# Mol-1 on AMD GPU (ROCm) — 安裝、patch 與 7pzb 範例

本分支是 [molence/Mol-1](https://github.com/molence/Mol-1) main 分支（mol-1-base）
加上 AMD ROCm 移植 patch。`constraint` 分支则是 mol-1-constraint（`mol cpred`）
的同款移植。所有結果皆在實體 AMD GPU 上實測。

## 測試環境（實測通過）

| 項目 | 版本 |
|---|---|
| GPU | 2x AMD Radeon AI PRO R9700（gfx1201 / RDNA4，各 32 GB） |
| OS | Ubuntu 24.04，amdgpu 核心驅動（ROCm 7.x 隨附） |
| Python | 3.13.9（venv） |
| PyTorch | 2.10.0+rocm7.0（HIP 7.0） |
| Triton | triton_rocm 3.6.0 |

> 注意：ROCm 不支援 CUDA-only 元件。安裝時**不要**裝 Mol-1 的 `[cuda]` extra
> （cuequivariance、fused LayerNorm 等都是 CUDA kernel），改用環境變數走
> PyTorch fallback（見下方）。

## 安裝

```bash
git clone <此 repo> && cd mol-1-amdgpu
python3.13 -m venv .venv-rocm
source .venv-rocm/bin/activate

# 1) ROCm 版 PyTorch（先裝，避免被一般 torch 蓋掉）
pip install torch --index-url https://download.pytorch.org/whl/rocm7.0

# 2) Mol-1 本體（不要加 [cuda] extra）
pip install -e .

# 3) 參考資料與 checkpoint（照 upstream README）
export MOL_ROOT_DIR="$PWD/data"
bash scripts/database/download_mol_data.sh          # ~0.3 GB
mkdir -p ${MOL_ROOT_DIR}/checkpoint
wget https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt \
  -O ${MOL_ROOT_DIR}/checkpoint/mol-1-base.pt       # ~2.2 GB
```

## ROCm patch（已包含在本 repo，共 3 個檔案）

1. `mol/model/tri_attention/op.py`
   Triton on ROCm 無法編譯 `tl.dot(input_precision="tf32")`。
   偵測到 `torch.version.hip` 時強制走 `deterministic=True`（kernel 內
   `input_precision="ieee"`，HIP 唯一支援的 precision）。推理路徑不受影響：
   推理沒有 backward，`dropout_p=0.0`。

2. `mol/model/tri_attention/autotune.py`
   Autotune 逐 config try/except：ROCm/RDNA 上部分 block/warp 組合編譯失敗，
   跳過該 config 而不是整輪 abort；全部失敗才 raise。

3. `mol/model/tri_attention/autotune_helpers.py`
   cache 目錄 `mkdir(exist_ok=True)`，避免多 rank race。

## 執行

```bash
export MOL_ROOT_DIR="$PWD/data"
export LAYERNORM_TYPE=torch            # 跳過 CUDA-only fused LayerNorm
CUDA_VISIBLE_DEVICES=0 mol pred -i examples/examples_with_template/example_9fm7.json \
  -o ./output --sample 1
```

必須帶的 CLI 選項（constraint 分支同理）：
`--triatt_kernel triattention --trimul_kernel torch`
（triangular multiplicative 的 Triton kernel 尚未移植，走 torch 實作。）

## 實測結果

| 輸入 | N_token | 時間 | peak VRAM | pLDDT | pTM | iTM |
|---|---|---|---|---|---|---|
| 9fm7（template+MSA） | 644 | 338 s | ~12 GB | 91.5 | 0.883 | 0.831 |
| MGYP004658859411 | — | — | — | 83.0 | 0.797 | 0* |
| 7pzb（本文件下方，完整 MSA） | 600 | 261 s | 11969 MiB | **NaN** | NaN | NaN |

\* MGYP 為單體，iTM 0 屬預期。

## ⚠ 7pzb 大 MSA 數值發散（重要發現，不限 AMD）

`examples/7pzb/`（N=600，paired 6017 + unpaired 6503 條 MSA）在 **bf16 與 fp32
都會在 recycle 中途（約 cycle 6–8）msa_module 內非有限發散**；推論本身會
「成功」但 summary 全為 NaN。舊版沒有 finite 檢查所以是靜默的；constraint 分支
（`mol cpred`）新增的 `NonFiniteTensorError` 只是把它抓出來。

已排除：autotune 選到的 kernel 差異（`--triatt_kernel torch` 同樣發散）、
LayerNorm 路徑、dropout（全 0）、dtype（fp32 亦發散）。

**解法（實測有效）**：把 inference 的 MSA 取樣深度壓小，例如
`data.msa.sample_cutoff.test = 2048`：

```bash
# 用 examples/amd_configs/msa_cutoff_2048.json（本 repo 內）
mol pred -i <input.json> -o ./out --config examples/amd_configs/msa_cutoff_2048.json \
  --triatt_kernel triattention --trimul_kernel torch
```

（`mol pred` 沒有 finite 檢查，跑完請務必確認 summary json 的 pLDDT 不是 NaN。）

## 已知限制 / 未做

- tri_muls 的 Triton kernel 未移植（走 `--trimul_kernel torch`）。
- fused LayerNorm（`LAYERNORM_TYPE=fused*`）是 CUDA-only，未移植。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` 在 HIP 不支援（只出現 warning）。
- bf16 autocast 下若仍遇到數值問題，可試 `-d fp32`（慢約 2–3 倍）。
