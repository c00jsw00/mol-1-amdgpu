# Mol-1 — AMD GPU (ROCm) 版

本 repo 是 [molence/Mol-1](https://github.com/molence/Mol-1) 的 **AMD ROCm 移植 fork**，
所有安裝與測試流程皆以 AMD GPU 為準。

- **main 分支** = mol-1-base（`mol pred`）
- **constraint 分支** = mol-1-constraint（`mol cpred`，binding-site 約束推論；見該分支 README）

數值發散分析、patch 細節與完整實測數據：[`AMD_ROCM.md`](AMD_ROCM.md)。

## 實測環境

| 項目 | 版本 |
|---|---|
| GPU | 2x AMD Radeon AI PRO R9700（gfx1201 / RDNA4，各 32 GB） |
| OS | Ubuntu 24.04（amdgpu 核心驅動，ROCm 7.x 隨附） |
| Python | 3.13（venv） |
| PyTorch | 2.10.0+rocm7.0（HIP 7.0） |
| Triton | triton_rocm 3.6.0 |

## 安裝（AMD）

```bash
git clone https://github.com/c00jsw00/mol-1-amdgpu && cd mol-1-amdgpu
python3 -m venv .venv-rocm && source .venv-rocm/bin/activate

# 1) 先裝 ROCm 版 PyTorch（避免被一般 torch 蓋掉）
pip install torch --index-url https://download.pytorch.org/whl/rocm7.0

# 2) Mol-1 本體（不要加 .[cuda] / .[cuda12] extra —— 那些是 NVIDIA 平台專用元件）
pip install -e .
```

## 資料與權重

```bash
export MOL_ROOT_DIR="$PWD/data"
bash scripts/database/download_mol_data.sh          # 參考資料 ~0.3 GB
mkdir -p ${MOL_ROOT_DIR}/checkpoint
wget https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt \
  -O ${MOL_ROOT_DIR}/checkpoint/mol-1-base.pt       # 權重 ~2.2 GB（或讓首次推論自動下載）
```

首次推論也可直接不带 `--load_checkpoint_path`，會自動下載到上述路徑；
加載是 strict 的（結構不符會直接拒絕）。

## 測試 example（9fm7，實測通過）

```bash
export LAYERNORM_TYPE=torch       # 跳過 NVIDIA 平台專用的 fused LayerNorm，走 PyTorch 實作
CUDA_VISIBLE_DEVICES=0 mol pred \
  -i examples/examples_with_template/example_9fm7.json \
  -o ./output --sample 1 \
  --triatt_kernel triattention --trimul_kernel torch
```

實測（單張 R9700）：338 s、peak ~12 GB VRAM、pLDDT 91.5 / pTM 0.883 / iTM 0.831。
結果在 `output/9fm7/seed_101/predictions/`（CIF 結構 + 每樣本一份 confidence JSON，
`sample_0` 是該 seed 排名最高的一組）。**跑完請確認 confidence JSON 的 pLDDT 不是 NaN。**

## 7pzb example（大 MSA，需附設定）

7pzb 的完整 MSA（12424 rows）會在中途 recycle 數值發散（bf16/fp32 皆然；
mol-1-base 與 mol-1-constraint 都一樣），推論會「成功」但結果全 NaN。
用 repo 內的設定把 inference MSA 取樣壓到 2048 rows 即可穩定跑通：

```bash
CUDA_VISIBLE_DEVICES=0 mol pred -i examples/7pzb_base_input.json \
  -o ./output --sample 1 \
  --config examples/amd_configs/msa_cutoff_2048.json \
  --triatt_kernel triattention --trimul_kernel torch
```

分析與對照實驗見 [`AMD_ROCM.md`](AMD_ROCM.md)。

## 蛋白質序列 + SMILES 配體 example（實測通過）

只有序列與配體 SMILES（無現成 MSA）的輸入：Mol-1 會自動跑 ColabFold MSA 搜尋。
`examples/prot_ligand_example/`（306 殘基蛋白質 + 一個 SMILES 配體）：

```bash
# 線上搜 MSA（走 ColabFold MMseqs2 公伺服）
CUDA_VISIBLE_DEVICES=0 mol pred -i examples/prot_ligand_example/input.json \
  -o ./output --sample 1 --triatt_kernel triattention --trimul_kernel torch

# 或離線：input_with_msa.json 已附 ./msa/non_pairing.a3m
CUDA_VISIBLE_DEVICES=0 mol pred -i examples/prot_ligand_example/input_with_msa.json \
  -o ./output --sample 1 --triatt_kernel triattention --trimul_kernel torch
```

實測（單張 R9700）：搜到 434 條 MSA、推論 93.75 s、peak 5,458 MiB、
pLDDT 96.4 / pTM 0.976 / ipTM 0.978。結果已存檔於
`examples/prot_ligand_example/predictions/`（CIF + confidence JSON）。

## 預測自己的輸入

輸入是一份 job 列表 JSON，每個 job 的 `sequences` 可含 `proteinChain`、
`dnaSequence`、`rnaSequence`、`ligand`、`ion`，可選 `covalent_bonds` 與殘基修飾；
MSA / template 檔可用 `pairedMsaPath` / `unpairedMsaPath` / `templatesPath`
直接附上。格式細節見 [docs/infer_json_format.md](docs/infer_json_format.md)
與 [`examples/`](examples/)。沒有附上的話 `mol prep` 會補齊 MSA 與 template
（MSA 走 ColabFold MMseqs2 公伺服，需要網路）。輸出欄位說明見
[docs/output_format.md](docs/output_format.md)。

使用 template 需要 `hmmer` 與 `kalign`（9fm7 範例自帶 template 結果，但套用
template 仍需要 kalign）：

```bash
sudo apt-get install hmmer kalign
mol pred -i <input.json> -o ./output --use_template true
```

## 指令一覽

| 指令 | 用途 |
| ---- | ---- |
| `mol pred` | 推論 |
| `mol json` | PDB/CIF 結構轉 Mol-1 JSON |
| `mol msa` | 蛋白質 MSA 搜尋 |
| `mol prep` | 補齊輸入缺的 MSA 與 template（RNA MSA 需 `--use_rna_msa true`） |
| `mol doctor` | 環境診斷（PyTorch build、GPU、kernel 選擇；提 bug 前先跑） |

各指令選項用 `mol <command> --help` 查看。

```bash
mol json --input examples/7pzb.pdb --out_dir ./output      # 結構轉輸入 JSON
mol msa  --input examples/prot.fasta --out_dir ./output    # 依 FASTA 逐筆找 MSA
mol prep --input examples/input.json --out_dir ./output    # 補 MSA 與 template
```

## AMD 注意事項

- 走 tri-attention kernel 時要帶 `--triatt_kernel triattention --trimul_kernel torch`
  （triangular-multiplicative 的 Triton kernel 尚未移植到 ROCm，走 torch 實作）。
- fused LayerNorm 是 NVIDIA 平台專用元件，務必 `LAYERNORM_TYPE=torch`。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` 在 HIP 不支援（只有 warning，沒有效果）。
- 遇到數值異常可試 `-d fp32`（慢約 2–3 倍，32 GB VRAM 下到 ~600 tokens 沒問題）。
- 顯存随 token 數近似二次方增長；大輸入的 chunk 控制見
  [docs/inference_memory.md](docs/inference_memory.md)（AMD 上未逐項實測，
  以每次推論 log 的 peak 用量為準）。
- 微調（fine-tune，`finetune_demo.sh`）流程未在本機 AMD 環境驗證，僅供參考：
  [docs/finetune.md](docs/finetune.md)。

## 文件

- [`AMD_ROCM.md`](AMD_ROCM.md) — patch 內容、數值發散分析、全部實測數據
- 上游文件：[Input JSON format](docs/infer_json_format.md) ·
  [Output format](docs/output_format.md) ·
  [MSA and template pipeline](docs/msa_template_pipeline.md) ·
  [Kernel options](docs/kernels.md) ·
  [Inference memory controls](docs/inference_memory.md) ·
  [Reference data](docs/data_sources.md)
- 上游專案與技術報告：[molence/Mol-1](https://github.com/molence/Mol-1)

## Acknowledgements

Mol-1 由 [Molence Bio](https://www.molence.com/) 開發；third-party attributions
and license information are provided in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
