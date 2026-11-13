# Mol-1-constraint — AMD GPU (ROCm) 版

本 repo 是 [molence/Mol-1](https://github.com/molence/Mol-1)（constraint 分支）的
**AMD ROCm 移植 fork**，所有安裝與測試流程皆以 AMD GPU 為準。

- **constraint 分支（本分支）** = mol-1-constraint（`mol cpred`，binding-site 約束推論）
- **main 分支** = mol-1-base（`mol pred`；見該分支 README）

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
git clone -b constraint https://github.com/c00jsw00/mol-1-amdgpu && cd mol-1-amdgpu
python3 -m venv .venv-rocm && source .venv-rocm/bin/activate

# 1) 先裝 ROCm 版 PyTorch（避免被一般 torch 蓋掉）
pip install torch --index-url https://download.pytorch.org/whl/rocm7.0

# 2) Mol-1-constraint 本體（不要加 .[cuda] / .[cuda12] extra —— 那些是 NVIDIA 平台專用元件）
pip install -e .
```

## 資料與權重

```bash
export MOL_ROOT_DIR="$PWD/data"
bash scripts/database/download_mol_data.sh          # 參考資料 ~0.3 GB
mkdir -p ${MOL_ROOT_DIR}/checkpoint
wget https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-constraint.pt \
  -O ${MOL_ROOT_DIR}/checkpoint/mol-1-constraint.pt # mol cpred 權重（缺檔也會自動下載）
wget https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt \
  -O ${MOL_ROOT_DIR}/checkpoint/mol-1-base.pt       # mol pred 權重（可選）
```

加載是 strict 的（model 分支、tensor 形狀與 checkpoint metadata 都要相符）。

## 測試 example（7pzb constraint binding site，實測通過）

```bash
export LAYERNORM_TYPE=torch       # 跳過 NVIDIA 平台專用的 fused LayerNorm，走 PyTorch 實作
CUDA_VISIBLE_DEVICES=0 mol cpred \
  -i examples/7pzb_constraint_binding_site/input.json \
  -o ./output --sample 1 \
  --config examples/amd_configs/dbg_msa_small.json \
  --triatt_kernel triattention --trimul_kernel torch
```

實測（單張 R9700）：666 s、peak 14.5 GB VRAM、pLDDT 89.7 / pTM 0.928 / iTM 0.928。
結果在 `output/7pzb_constraint_binding_site/seed_101/predictions/`；實測 confidence
JSON 已存檔於 `examples/results_7pzb_amd/cpred_msa2048_summary_confidence.json`。
**跑完請確認 confidence JSON 的 pLDDT 不是 NaN。**

**為什麼要附 `--config`**：7pzb 的完整 MSA（12424 rows）在 mol-1-base 與
mol-1-constraint 都會在 recycle 中途數值發散（bf16/fp32、triattention 或 torch
kernel 皆然），推論會「成功」但結果全 NaN。`examples/amd_configs/dbg_msa_small.json`
把 inference MSA 取樣壓到 2048 rows 即可穩定跑通；分析見 `AMD_ROCM.md`。

其他可用設定：
- `examples/amd_configs/dbg_numerics.json` — 開啟每個 submodule 的 amax log，
  配合 `NonFiniteTensorError` 定位發散點。
- `--use_msa False` — 完全不用 MSA 也可跑通（品質較低：pLDDT 84.6 / iTM 0.22）。
- `examples/amd_configs/7pzb_base_input.json` — 不帶 constraint 的對照輸入（給 `mol pred`）。

## 使用 constraint

在輸入 JSON 加 SITE、CONTACT 或 EXCLUSION（語義見
[docs/constraint_semantics.md](docs/constraint_semantics.md)）；三種皆可混合使用。
constraint 只是引導推論，不保證滿足、也不會篩掉候選。

推論預設 20 recycles、200 diffusion steps，可用 `--cycle` / `--step` 覆寫；
注意 `--use_default_params true` 會強制 20/200。一般 CLI 選項優先於 `--config` JSON。

## 蛋白質序列 + SMILES 配體 + SITE constraint example（實測通過）

`examples/prot_ligand_site_example/`：306 殘基蛋白質 + SMILES 配體，SITE
constraint 指定 A 鏈殘基 41, 49, 142–145 為配體結合位點。

```bash
# 離線版（附 ./msa/non_pairing.a3m，434 rows，不必再附 MSA 取樣設定）
CUDA_VISIBLE_DEVICES=0 mol cpred -i examples/prot_ligand_site_example/input.json \
  -o ./output --sample 1 --triatt_kernel triattention --trimul_kernel torch

# 或線上搜 MSA
CUDA_VISIBLE_DEVICES=0 mol cpred -i examples/prot_ligand_site_example/input_search_msa.json \
  -o ./output --sample 1 --triatt_kernel triattention --trimul_kernel torch
```

實測（單張 R9700）：推論 74.6 s、peak 5,458 MiB、pLDDT 95.0 / pTM 0.971 /
ipTM 0.947。constraint 驗證：六個指定殘基皆與配體形成接觸（3.0–3.7 Å），
配體完整接觸譜為殘基 41, 49, 140–145, 163–168, 172, 187–191。結果已存檔於
`examples/prot_ligand_site_example/predictions/`（CIF + confidence JSON）。
本例 MSA 只有 434 rows，遠低於會發散的量級，不需要 `--config` 取樣設定。

## 預測自己的輸入

輸入是一份 job 列表 JSON，每個 job 的 `sequences` 可含 `proteinChain`、
`dnaSequence`、`rnaSequence`、`ligand`、`ion`，可選 `covalent_bonds` 與殘基修飾；
MSA / template 檔可用 `pairedMsaPath` / `unpairedMsaPath` / `templatesPath`
直接附上。格式細節見 [docs/infer_json_format.md](docs/infer_json_format.md)
與 [`examples/`](examples/)。沒有附上的話 `mol prep` / `mol cpred` 會補齊
（MSA 走 ColabFold MMseqs2 公伺服，需要網路）。輸出欄位說明見
[docs/output_format.md](docs/output_format.md)。

使用 template 需要 `hmmer` 與 `kalign`：

```bash
sudo apt-get install hmmer kalign
mol cpred -i <input.json> -o ./output --use_template true
```

另有 beta 的 evidence-to-constraints 輔助 skill（把實驗證據轉成 constraint 輸入，
非必要）：[docs/mol1-evidence-to-constraints/SKILL.md](docs/mol1-evidence-to-constraints/SKILL.md)。

## 指令一覽

| 指令 | 用途 |
| ---- | ---- |
| `mol cpred` | Mol-1-constraint 推論（本分支主指令） |
| `mol pred` | 基礎模型推論（需 mol-1-base 權重） |
| `mol json` | PDB/CIF 結構轉 Mol-1 JSON |
| `mol msa` | 蛋白質 MSA 搜尋 |
| `mol prep` | 補齊輸入缺的 MSA 與 template（RNA MSA 需 `--use_rna_msa true`） |
| `mol doctor` | 環境診斷（PyTorch build、GPU、kernel 選擇；提 bug 前先跑） |

各指令選項用 `mol <command> --help` 查看。

```bash
mol json --input examples/7pzb.pdb --out_dir ./output      # 結構轉輸入 JSON
mol msa  --input examples/prot.fasta --out_dir ./output    # 依 FASTA 逐筆找 MSA
mol prep --input examples/input.json --out_dir ./output    # 補 MSA 與 template
bash inference_demo.sh                                     # 7PZB 範例的腳本版
```

## AMD 注意事項

- 務必帶 `--triatt_kernel triattention --trimul_kernel torch`
  （triangular-multiplicative 的 Triton kernel 尚未移植到 ROCm，走 torch 實作）。
- fused LayerNorm 是 NVIDIA 平台專用元件，務必 `LAYERNORM_TYPE=torch`。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` 在 HIP 不支援（只有 warning，沒有效果）。
- 遇到數值異常可試 `-d fp32`（慢約 2–3 倍，32 GB VRAM 下到 ~600 tokens 沒問題）。
- 顯存随 token 數近似二次方增長；chunk 控制見
  [docs/inference_memory.md](docs/inference_memory.md)（AMD 上未逐項實測，
  以每次推論 log 的 peak 用量為準）。
- 微調 / 訓練流程未在本機 AMD 環境驗證，僅供參考：
  [docs/finetune.md](docs/finetune.md)。

## Benchmarks（上游結果，非 AMD 實測）

**與各 constraint 模型比較**：Mol-1-constraint、Chai-1、Boltz-1、Boltz-2、
Protenix-v0.5 皆以參考結合位點（interface residues）作為 constraint 輸入。
左圖為 160 個 FoldBench 抗體–抗原界面（epitope，success DockQ >0.23 與 >0.80）；
右圖為 511 個蛋白質–配體界面（pocket，standard success 需 iLDDT ≥0.8 且
ligand RMSD ≤2 Å；strict success 需同候選 ≥0.9 且 ≤1.5 Å）。以 5 seeds ×
每 seed 5 samples 的 chain-pair ipTM top-1 評選。

<img src="assets/images/constraint-benchmark-overview.png" style="width: 95%; height: auto;" alt="Constraint model comparisons on antibody–antigen and protein–ligand interfaces">

**對不完整／錯誤 site 輸入的穩健性**：五個 constraint 模型共用 150 個
抗體–抗原界面，依 chain-pair ipTM 評選。由左至右：正確 site 覆盖率递增；
在完整參考 site 加入錯誤殘基；在 60% 參考 site 加入錯誤殘基。曲線為
參考結合位點 contact recovery（CSR），虛線為 Mol-1 對照。

<img src="assets/images/constraint-robustness-overview.png" style="width: 95%; height: auto;" alt="Reference-site CSR under increasing correct coverage and two incorrect-site perturbations">

## 文件

- [`AMD_ROCM.md`](AMD_ROCM.md) — patch 內容、數值發散分析、全部實測數據
- 上游文件：[Constraint semantics](docs/constraint_semantics.md) ·
  [Input JSON format](docs/infer_json_format.md) ·
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
