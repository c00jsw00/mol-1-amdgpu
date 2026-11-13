# Mol-1-constraint

<img src="assets/images/ribbon-mint-hotspots.png" style="width: 100%; height: auto;" alt="Mol-1 overview">




<h3 align="center">
  <strong>Mol-1:</strong> Open Frontier Biomolecular Intelligence
</h3>

<p align="center">
  <a href="assets/report.pdf">📄 Mol-1 Technical report</a> ·
  <a href="https://lab.molence.com/">🧪 Molence Lab (Platform)</a> ·
  by <a href="https://www.molence.com/"><img src="assets/images/molence-icon.svg" width="18" height="12" alt=""> Molence Bio</a>
</p>



<div align="center">

[![X](https://img.shields.io/badge/X.com-Follow-black?logo=x)](https://x.com/MolenceBio) 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Follow-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/company/molence-bio/)
[![Slack](https://img.shields.io/badge/Slack-Join-4A154B?logo=slack)](https://join.slack.com/t/molence/shared_invite/zt-4ef45vmn5-X9~YKJRQFhb8zEkZ0HP9Iw)
[![Email](https://img.shields.io/badge/Email-Contact-orange?logo=gmail)](mailto:contact@molence.com) \
[![Wechat Official Account](https://img.shields.io/badge/Wechat_Official_Account-Follow-brightgreen?logo=wechat)](assets/images/wechat-official-account.png)
[![Wechat](https://img.shields.io/badge/Wechat_Group-Join-brightgreen?logo=wechat)](assets/images/wechat-group-3.png)

</div>





## ⚡ AMD GPU (ROCm) — 安裝與測試

本分支（constraint，`mol cpred`）已移植到 AMD ROCm，patch 已在 repo 內；
`main` 分支是 mol-1-base 同款移植。完整說明：[`AMD_ROCM.md`](AMD_ROCM.md)。

實測環境：2x AMD Radeon AI PRO R9700（gfx1201 / RDNA4，32 GB）、Ubuntu 24.04、
Python 3.13、PyTorch 2.10.0+rocm7.0、triton_rocm 3.6.0。

### 安裝（AMD）

```bash
git clone -b constraint https://github.com/c00jsw00/mol-1-amdgpu && cd mol-1-amdgpu
python3 -m venv .venv-rocm && source .venv-rocm/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/rocm7.0  # 先裝 ROCm 版 torch
pip install -e .            # 注意：AMD 不要裝 ".[cuda]"（CUDA-only 元件）

export MOL_ROOT_DIR="$PWD/data"
bash scripts/database/download_mol_data.sh          # 參考資料 ~0.3 GB
mkdir -p ${MOL_ROOT_DIR}/checkpoint
wget https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-constraint.pt \
  -O ${MOL_ROOT_DIR}/checkpoint/mol-1-constraint.pt # mol cpred 權重
wget https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt \
  -O ${MOL_ROOT_DIR}/checkpoint/mol-1-base.pt       # mol pred 權重（可選）
```

### 測試 example（7pzb constraint binding site，實測通過）

```bash
export LAYERNORM_TYPE=torch       # 跳過 CUDA-only fused LayerNorm
CUDA_VISIBLE_DEVICES=0 mol cpred \
  -i examples/7pzb_constraint_binding_site/input.json \
  -o ./output --sample 1 \
  --config examples/amd_configs/dbg_msa_small.json \
  --triatt_kernel triattention --trimul_kernel torch
```

實測：666 s、peak 14.5 GB VRAM、pLDDT 89.7 / pTM 0.928 / iTM 0.928
（JSON 輸出存檔於 `examples/results_7pzb_amd/cpred_msa2048_summary_confidence.json`）。

**為什麼要附 `--config`**：7pzb 的完整 MSA（12424 rows）在 mol-1-base 與
mol-1-constraint 都會在 recycle 中途數值發散（bf16/fp32、triattention 或 torch
kernel 皆然），推論會「成功」但結果全 NaN。`examples/amd_configs/dbg_msa_small.json`
把 inference MSA 取樣壓到 2048 rows 即可穩定跑通；分析見 `AMD_ROCM.md`。

其他可用設定：
- `examples/amd_configs/dbg_numerics.json` — 開啟每個 submodule 的 amax log，
  配合 `NonFiniteTensorError` 定位發散點。
- `--use_msa False` — 完全不用 MSA 也可跑通（品質較低：pLDDT 84.6 / iTM 0.22）。
- `examples/amd_configs/7pzb_base_input.json` — 不帶 constraint 的對照輸入（給 `mol pred`）。

### AMD 注意事項

- 務必帶 `--triatt_kernel triattention --trimul_kernel torch`
  （triangular-multiplicative Triton kernel 尚未移植）。
- fused LayerNorm 是 CUDA-only，務必 `LAYERNORM_TYPE=torch`。
- 數值異常時可試 `-d fp32`（慢 2–3 倍）。

## Install

You can install `Mol-1-constraint` with the following commands:

```bash
git clone --branch constraint https://github.com/molence/Mol-1.git
cd Mol-1

pip install -e ".[cuda]"      # NVIDIA GPU
# or
pip install -e ".[cpu]"       # CPU
```

If you are on a CUDA driver whose version is prior to CUDA 13.0, install with `pip install -e ".[cuda12]"` instead.

> The first prediction or training run on a CUDA machine compiles the fused LayerNorm CUDA extension: a couple of minutes with no output but compiler lines, cached per environment afterwards, so a first run that seems to hang is usually still compiling.

## Quick-start

```bash
export MOL_ROOT_DIR="./data"    # model weight and reference data will be put here

bash scripts/database/download_mol_data.sh   # run once: reference data, ~0.3 GB

mkdir -p ${MOL_ROOT_DIR}/checkpoint
# download Mol-1-constraint weights (also downloaded automatically if missing)
wget https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-constraint.pt \
  -O "${MOL_ROOT_DIR}/checkpoint/mol-1-constraint.pt"

# Optional base weights for Training from base below:
# wget https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt \
#   -O "${MOL_ROOT_DIR}/checkpoint/mol-1-base.pt"

# predict a protein–DNA–ligand complex (600 tokens) 
# with three protein residues as binding site 
mol cpred -i examples/7pzb_constraint_binding_site/input.json -o ./output
```

Results land in `output/7pzb_constraint_binding_site/seed_101/predictions/` as 5 CIF structures and their confidence JSON files. Templates are off by default;
see [Template search and align](#template-search-and-align) to enable them.


## Predict with Mol-1-constraint
The easiest way to run Mol-1-constraint prediction is:

```bash
mol cpred -i /path/to/input/json -o /path/to/output/dir
```

**Input format**. `Mol-1-constraint`'s input is a list of "jobs" described by JSON, where each job's `sequences` holds `proteinChain`, `dnaSequence`, `rnaSequence`, `ligand` or `ion` entries, with optional
`covalent_bonds` and residue modifications. Template hits are accepted as
`.a3m`, `.hhr` or JSON. See the details in [input JSON format](docs/infer_json_format.md) and
[`examples/`](examples/), which includes a nanobody case driven by JSON
templates.

**Constraints**. Add SITE, CONTACT or EXCLUSION to the input JSON as described in
[constraint semantics](docs/constraint_semantics.md). The released model supports
all three; each input may use any combination or none. Conditioning guides the
prediction without guaranteeing satisfaction or filtering candidates.

**Optional agent skill (beta).** The
[evidence-to-constraints skill](docs/mol1-evidence-to-constraints/SKILL.md) is a
companion tool for converting experimental observations (e.g., from literature) into
proposed Mol-1 constraint inputs, with source evidence, residue/token mapping, and technical
validation. It is **not required** to use the model. Users are responsible for verifying
the authenticity of the evidence and assessing its scientific interpretation.

Inference defaults to 20 recycles and 200 diffusion steps. Override them with
`--cycle` and `--step`. Note that setting `--use_default_params true` forces 20/200 even when other
budgets are supplied. Other explicit CLI options override `--config` JSON values.

**Prediction outputs**. Results are organized in `output/<job_name>/seed_<seed>/predictions/` as CIF structures with a confidence JSON per sample. Within each seed, samples are numbered by `ranking_score`, highest first, so `sample_0` is that seed's top-ranked structure.

**Confidence scores**. [docs/output_format.md](docs/output_format.md) describes every
field, including two that trip people up — `plddt` is scaled 0–100 while
`chain_plddt` is 0–1, and the PAE matrix is indexed by token rather than by
residue number. The per-token and per-atom data, the PAE matrix among them, go
to a separate `full_data` JSON written only with `--need_atom_confidence true`.

**MSA and template**. For an input of your own, `mol cpred` searches for what the JSON does not carry. 
Protein MSAs come from the public ColabFold MMseqs2 service, so
`--use_msa`, on by default, needs network access. With `--use_template true`,
templates are searched with HMMER against the `search_database/` archive,
starting from the chain's MSA. A `pairedMsaPath`, `unpairedMsaPath` or
`templatesPath` already in the JSON skips the matching search. The
[MSA and template pipeline](docs/msa_template_pipeline.md) describes the files.

**RNA MSAs**. RNA MSAs are off by default. With `--use_rna_msa true`, RNA chains without an
`unpairedMsaPath` are searched with HMMER against three further sequence
databases (Rfam, RNAcentral and an NCBI `nt` RNA subset, tens of gigabytes) that
the release archives do not include; [docs/data_sources.md](docs/data_sources.md)
names them.

**Script example**. `bash inference_demo.sh` runs the prepared
[7PZB binding-site example](examples/7pzb_constraint_binding_site/input.json)
through `mol/constraint/inference.py`, reusing its MSAs with templates disabled.
Set `CHECKPOINT_PATH` to use custom weights; additional native runner options
can be appended to the script command.

### Template search and align
The example `examples/examples_with_template/example_9fm7.json` carries precomputed MSAs and template hits, but using templates still needs `kalign`, which aligns every hit to the query; the hit structures are downloaded from PDBe on first use. Template search for your own inputs also needs HMMER. Install both:

```bash
sudo apt-get install hmmer kalign

# or, in a conda environment (keep the kalign3 pin: later releases align differently)
conda install -c conda-forge -c bioconda --override-channels hmmer kalign3=3.3.5

# predict example with template
mol cpred -i examples/examples_with_template/example_9fm7.json -o ./output \
  --use_template true
```


## Commands

`mol` has these commands:

| Command | What it does |
| ------- | ------------ |
| `mol cpred` | Run Mol-1-constraint inference |
| `mol pred` | Run base-model inference with base weights |
| `mol json` | Convert PDB/CIF structures to Mol-1 JSON |
| `mol msa` | Protein MSA search |
| `mol prep` | MSA and template search; RNA MSA search with `--use_rna_msa true` |
| `mol doctor` | Environment diagnostics |

Use `mol <command> --help` to list the options for each.

For example:

```bash
mol json --input examples/7pzb.pdb --out_dir ./output      # structure to input JSON
mol msa --input examples/prot.fasta --out_dir ./output     # MSAs per FASTA record, listed in msa_index.json
mol prep --input examples/input.json --out_dir ./output    # MSAs and templates an input lacks
```

## Weights and data

Quick-start fetches everything prediction needs. For fine-tuning,
`bash scripts/database/download_mol_data.sh --full` also fetches the
precomputed MSAs and template hits (9.8 GB, 46 GB extracted).
[docs/data_sources.md](docs/data_sources.md) lists every archive with its
contents and download link.

`mol cpred` uses `$MOL_ROOT_DIR/checkpoint/mol-1-constraint.pt`, downloading the
[released Mol-1-constraint weights](https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-constraint.pt)
there if missing. Pass `--load_checkpoint_path /path/to/mol-1-constraint.pt` to
use custom weights; an explicit path must exist and is never replaced by a download.
Regular and EMA exports both load the supplied `model` tensors directly.
Loading is strict: model branches, tensor shapes and checkpoint metadata must match.
Base weights for training are available from the commented download command in
[Quick-start](#quick-start).


## Fine-tune Mol-1

You may find it useful to fine-tune Mol-1 by building a training dataset from your own **mmCIF** files. The repository includes a **six-structure dataset** to try it on:

```bash
# convert structures -> bioassembly dicts
mkdir -p finetune_data
ls scripts/msa/data/mmcif/*.cif examples/finetune/*.cif > finetune_data/structures.txt
# prepare chain/interface index table
python3 scripts/prepare_training_data.py -i finetune_data/structures.txt \
  -b finetune_data/bioassembly -o finetune_data/indices.csv \
  -c "$MOL_ROOT_DIR/common/clusters-by-entity-40.txt"

# prepare protein MSAs, through the same search service as inference (recommended)
python3 scripts/prepare_finetune_msa.py -b finetune_data/bioassembly \
  -o finetune_data/msa
```

The fine-tuning script reads `FINETUNE_DATA_DIR` (default `./finetune_data`).

Fine-tune base weights:
```bash
CHECKPOINT_PATH="${MOL_ROOT_DIR}/checkpoint/mol-1-base.pt" bash finetune_demo.sh \
  --max_steps 6 --checkpoint_interval 3
```

Dataset preparation and base fine-tuning options are described in
[docs/finetune.md](docs/finetune.md).


## Environment diagnosis

`mol doctor` reports the PyTorch build, the GPUs it sees, which kernels `auto`
picks, the preprocessing binaries and the data it finds. It is read-only and
loads no model; run it before filing a bug.
[docs/installation.md](docs/installation.md#verify) shows a report and what to
look for in it.


## Benchmarks

**Comparison with constraint models.** Mol-1-constraint, Chai-1, Boltz-1,
Boltz-2 and Protenix-v0.5 receive reference-derived binding-site (interface residue) constraint as input.
The left panel covers 160 FoldBench antibody–antigen interfaces (epitope, success DockQ >0.23
and >0.80); the right covers 511 protein–ligand interfaces (pocket, standard success requires iLDDT ≥0.8 and ligand RMSD ≤2 Å; strict success requires ≥0.9 and ≤1.5 Å in the same candidate). Results select the top-1 candidate by chain-pair ipTM from 5 seeds with 5 samples per seed. This comparison measures how models use it.

<img src="assets/images/constraint-benchmark-overview.png" style="width: 95%; height: auto;" alt="Constraint model comparisons on antibody–antigen and protein–ligand interfaces">

**Robustness to incomplete and incorrect site inputs.** All five constraint models share 150 antibody–antigen interfaces, ranked by chain-pair
ipTM. From left to right: increasing correct-site coverage; adding incorrect
residues to the full reference site; and adding incorrect residues to a 60%
reference site. Curves show reference binding-site contact recovery (CSR); the dashed line is Mol-1 performance for comparison.

<img src="assets/images/constraint-robustness-overview.png" style="width: 95%; height: auto;" alt="Reference-site CSR under increasing correct coverage and two incorrect-site perturbations">


## Documentation

- [Installation](docs/installation.md)
- [Fine-tuning on your own structures](docs/finetune.md)
- [Preparing training data](docs/prepare_training_data.md)
- [Input JSON format](docs/infer_json_format.md)
- [Constraint semantics and input examples](docs/constraint_semantics.md)
- [Evidence-to-constraints skill](docs/mol1-evidence-to-constraints/SKILL.md)
- [Output format](docs/output_format.md)
- [Kernel options](docs/kernels.md)
- [Inference memory controls](docs/inference_memory.md)
- [Reference data](docs/data_sources.md)
- [MSA and template pipeline](docs/msa_template_pipeline.md)

## Acknowledgements

Third-party attributions and license information are provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). 


## Citation

Coming soon :)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
