---
name: mol-1-use
description: Use when predicting protein/ligand complexes with Mol-1 (mol pred / mol cpred) on AMD ROCm.
---

# Mol-1 使用技能（mol-1-base + mol-1-constraint，AMD ROCm）

Use when: 要用 Mol-1 預測蛋白質（含配體、多鏈）結構，或用 constraint 版本
指定結合位點預測；環境為 AMD GPU（ROCm）。兩個分支：

| 分支 | checkpoint | CLI | 用途 |
|---|---|---|---|
| main (mol-1-base) | mol-1-base.pt | `mol pred` | 一般結構預測（蛋白質/配體/抗體/核酸） |
| constraint (mol-1-constraint) | mol-1-constraint.pt | `mol cpred` | 帶 constraint（SITE 結合位點等）的預測 |

## 環境前置（每次都要）

```bash
cd <repo>                      # Mol-1 (main) 或 Mol-1-constraint (constraint)
export MOL_ROOT_DIR="$PWD/data"   # checkpoint 與 MSA 資料庫根目錄
export LAYERNORM_TYPE=torch       # 跳過 CUDA-only fused LayerNorm
export CUDA_VISIBLE_DEVICES=0     # ROCm PyTorch 也用這個變數選卡
```

AMD 執行必帶旗標（CLI 選項不是 `--triangle_attention`）：

```bash
--triatt_kernel triattention --trimul_kernel torch
```

禁止：`pip install ".[cuda]"` / `.[cuda12]`（會拉 CUDA-only 的
cuequivariance / fused kernels，HIP 上裝不起來）。`PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` 在 HIP 不支援，只出現 warning，不要加。

## mol-1-base：mol pred

```bash
mol pred -i input.json -o ./output --sample 1 \
  --triatt_kernel triattention --trimul_kernel torch
```

輸入 JSON（頂層是 list，每個 object 一個預測 job）：

```json
[{ "name": "job1",
   "sequences": [
     {"proteinChain": {"sequence": "SGFRK...", "count": 1}},
     {"ligand": {"ligand": "CCC(C)C(C(=O)N...)C=O", "count": 1}}
   ]}]
```

- `ligand.ligand` 是 SMILES；`count` 是拷貝數。
- 沒給 MSA 路徑時會自動線上搜 ColabFold MSA（需網路）。
- 離線重跑：跑完會產生 `<name>-update-msa.json`，或在 proteinChain 加
  `"unpairedMsaPath": "./path/non_pairing.a3m"`（必要時 `pairingMsaPath`）。

輸出：`output/<name>/seed_101/predictions/<name>_sample_0.cif` +
`<name>_summary_confidence_sample_0.json`（pLDDT / pTM / ipTM /
ranking_score / has_clash）。seed 目錄固定 seed_101（預設種子）。

## mol-1-constraint：mol cpred

```bash
mol cpred -i input.json -o ./output --sample 1 \
  --triatt_kernel triattention --trimul_kernel torch
```

輸入 JSON 同上，另加 `id` 與 `constraints`：

```json
[{ "name": "job1",
   "sequences": [
     {"proteinChain": {"sequence": "...", "count": 1, "id": ["A"],
       "unpairedMsaPath": "./msa/non_pairing.a3m"}},
     {"ligand": {"ligand": "SMILES...", "count": 1, "id": ["B"]}}
   ],
   "constraints": {
     "site": {
       "source_chains": ["A"],
       "partner_chains": ["B"],
       "residue_ids": [["A",41], ["A",49], ["A",142], ["A",143], ["A",144], ["A",145]]
     }}}]
```

SITE constraint 語意：把 `residue_ids` 指定的鏈/殘基標為與
`partner_chains`（配體鏈）的結合位點。預測後可用 CIF 距離驗證：指定殘基
應與配體重原子形成 <4 Å 接觸。

## 驗證預測品質

1. confidence JSON 有 NaN → 數值發散，見下方 pitfall，結果不可用。
2. `has_clash: false` 且 pLDDT > 90、ipTM > 0.9 通常可信。
3. constraint 是否生效：解析 CIF `_atom_site`（照 header 欄位名解析，
   勿假設固定欄位序），算 source 殘基與配體最小距離。
4. 有實驗結構時用 RMSD：共有的 CA 做 Kabsch 疊合；配體用 RDKit
   次結構圖論配對（設 `BondType.ZERO` 只比連接性），可參照
   `examples/prot_ligand_site_example/compare_with_7wof.py`。

## Pitfalls（AMD / 數值）

- 大 MSA（>~6000 rows，如 7pzb）在 bf16 與 fp32 都會在 cycle 6–8 後
  non-finite（`NonFiniteTensorError` 或 confidence 全 NaN 卻報
  succeeded）。解法：`--config` 降低取樣深度，
  `data.msa.sample_cutoff.test = 2048` + `model.msa_max_size = 2048`
  （見 `examples/amd_configs/`）。不要用 head 截 a3m 大檔。
- MSA 小（數百 rows）不需要任何 config 修正。
- `mol pred` 報 succeeded 不等於成功：務必檢查 summary JSON 有無 NaN。
- triattention autotune 已打補丁（IEEE 精度、autotune 容錯、cache mkdir
  race）；改動在 `mol/model/tri_attention/{autotune,autotune_helpers,op}.py`。
- MSA 預處理可能印出 conda 偵測 `TypeError: ... NoneType`，不影響結果；
  以 exit code + 輸出檔存在與否判定。
- 推論 >7 min 的 job 用背景程序跑，避免前端逾時。

## 實測基準（單張 AMD Radeon AI PRO R9700, 32GB）

| 案例 | 時間 | peak VRAM | pLDDT / pTM / ipTM |
|---|---:|---:|---|
| 306aa Mpro + GC376, mol pred, MSA 434 | 93.7 s | 5.4 GB | 96.4 / 0.976 / 0.978 |
| 同上 + SITE constraint, mol cpred | 74.6 s | 5.4 GB | 95.0 / 0.971 / 0.947 |
| 7pzb, cpred, MSA cutoff 2048 | 666 s | 14.5 GB | 89.7 / 0.928 / 0.928 |
| 7pzb, cpred, full MSA | — | — | NaN（不可用） |

vs 實驗結構 7WOF：CA RMSD base 0.71 Å / constraint 0.69 Å；
配體 RMSD（蛋白質疊合下）1.28 Å / 0.66 Å — constraint 明顯改善配體擺位。

## Checkpoints

```
https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt
https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-constraint.pt
```

放到 `$MOL_ROOT_DIR/checkpoint/`。
