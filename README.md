# Mol-1

<img src="assets/images/ribbon-mint-portrait.png" style="width: 100%; height: auto;" alt="Mol-1 overview">




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

本 fork 已把 Mol-1 移植到 AMD ROCm（patch 已在 repo 內）。
**main 分支** = mol-1-base（`mol pred`）；**constraint 分支** = mol-1-constraint
（`mol cpred`，見該分支 README / `AMD_ROCM.md`）。完整說明：[`AMD_ROCM.md`](AMD_ROCM.md)。

實測環境：2x AMD Radeon AI PRO R9700（gfx1201 / RDNA4，32 GB）、Ubuntu 24.04、
Python 3.13、PyTorch 2.10.0+rocm7.0、triton_rocm 3.6.0。

### 安裝（AMD）

```bash
git clone https://github.com/c00jsw00/mol-1-amdgpu && cd mol-1-amdgpu
python3 -m venv .venv-rocm && source .venv-rocm/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/rocm7.0  # 先裝 ROCm 版 torch
pip install -e .            # 注意：AMD 不要裝 ".[cuda]"（CUDA-only 元件）

export MOL_ROOT_DIR="$PWD/data"
bash scripts/database/download_mol_data.sh          # 參考資料 ~0.3 GB
mkdir -p ${MOL_ROOT_DIR}/checkpoint
wget https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt \
  -O ${MOL_ROOT_DIR}/checkpoint/mol-1-base.pt       # 權重 ~2.2 GB（或讓首次推論自動下載）
```

### 測試 example（9fm7，實測通過）

```bash
export LAYERNORM_TYPE=torch       # 跳過 CUDA-only fused LayerNorm
CUDA_VISIBLE_DEVICES=0 mol pred \
  -i examples/examples_with_template/example_9fm7.json \
  -o ./output --sample 1 \
  --triatt_kernel triattention --trimul_kernel torch
```

實測：338 s、peak ~12 GB VRAM、pLDDT 91.5 / pTM 0.883 / iTM 0.831
（結果存 `output/9fm7/seed_101/predictions/..._summary_confidence_sample_0.json`，
跑完請確認 pLDDT 不是 NaN）。

### 7pzb example（大 MSA，需附設定）

完整 MSA（12424 rows）在 mol-1-base 與 mol-1-constraint 都會數值發散（bf16/fp32
皆然，推論「成功」但 summary 全 NaN——見 `AMD_ROCM.md` 的分析）。
用 repo 內的 MSA 取樣設定即可跑通：

```bash
CUDA_VISIBLE_DEVICES=0 mol pred -i examples/7pzb_base_input.json \
  -o ./output --sample 1 \
  --config examples/amd_configs/msa_cutoff_2048.json \
  --triatt_kernel triattention --trimul_kernel torch
```

### AMD 注意事項

- 自訂輸入走 tri-attention kernel 時要帶
  `--triatt_kernel triattention --trimul_kernel torch`。
- fused LayerNorm 是 CUDA-only，務必 `LAYERNORM_TYPE=torch`。
- 數值異常時可試 `-d fp32`（慢 2–3 倍）。

## Install

You can install `Mol-1` with the following commands:

```bash
git clone https://github.com/molence/Mol-1 && cd Mol-1

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
# download model checkpoint, ~2.2 GB
wget https://mol-1-release.oss-cn-shanghai.aliyuncs.com/checkpoint/mol-1-base.pt \
  -O ${MOL_ROOT_DIR}/checkpoint/mol-1-base.pt

# predict example: a homodimer with two ligands (644 tokens)
mol pred -i examples/examples_with_template/example_9fm7.json -o ./output
```

The example needs about 14 GB of GPU memory; on a 95 GiB NVIDIA Blackwell GPU it takes about two minutes, half of it loading the model, plus a few minutes the first time for the CUDA build above. If it goes well, results will land in `output/9fm7/seed_101/predictions/` as CIF structures with a confidence JSON per sample. By default, Mol-1 predicts without templates (`--use_template false`); to use them, install the extra dependencies described in [Template search and align](#template-search-and-align).


## Predict with Mol-1
The easiest way to run Mol-1 prediction is:

```bash
mol pred -i /path/to/input/json -o /path/to/output/dir
```

**Input format**. `Mol-1`'s input is a list of "jobs" described by JSON, where each job's `sequences` holds `proteinChain`, `dnaSequence`, `rnaSequence`, `ligand` or `ion` entries, with optional
`covalent_bonds` and residue modifications. Template hits are accepted as
`.a3m`, `.hhr` or JSON. See the details in [input JSON format](docs/infer_json_format.md) and
[`examples/`](examples/), which includes a nanobody case driven by JSON
templates.

**Prediction outputs**. Results are organized in `output/<job_name>/seed_<seed>/predictions/` as CIF structures with a confidence JSON per sample. Within each seed, samples are numbered by `ranking_score`, highest first, so `sample_0` is that seed's top-ranked structure.

**Confidence scores**. [docs/output_format.md](docs/output_format.md) describes every
field, including two that trip people up — `plddt` is scaled 0–100 while
`chain_plddt` is 0–1, and the PAE matrix is indexed by token rather than by
residue number. The per-token and per-atom data, the PAE matrix among them, go
to a separate `full_data` JSON written only with `--need_atom_confidence true`.

**MSA and template**. For an input of your own, `mol pred` searches for what the JSON does not carry. Protein MSAs come from the public ColabFold MMseqs2 service, so
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

**Script example**. [`inference_demo.sh`](inference_demo.sh) runs the same
*9fm7* example through `mol/runner/inference.py`, which accepts every config
setting as an option; the comment at the top of the script explains its use.

### Template search and align
The example `examples/examples_with_template/example_9fm7.json` carries precomputed MSAs and template hits, but using templates still needs `kalign`, which aligns every hit to the query; the hit structures are downloaded from PDBe on first use. Template search for your own inputs also needs HMMER. Install both:

```bash
sudo apt-get install hmmer kalign

# or, in a conda environment (keep the kalign3 pin: later releases align differently)
conda install -c conda-forge -c bioconda --override-channels hmmer kalign3=3.3.5

# predict example with template
mol pred -i examples/examples_with_template/example_9fm7.json -o ./output \
  --use_template true
```


## Commands

`mol` has these commands:

| Command | What it does |
| ------- | ------------ |
| `mol pred` | Run inference |
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

Pass the weights to `mol pred` with `--load_checkpoint_path`, or put them at `$MOL_ROOT_DIR/checkpoint/mol-1-base.pt`; when neither is there, the first prediction downloads them to that path. An explicit path is always used as given. Loading is strict: a checkpoint whose shapes do not match the architecture is rejected rather than partially loaded. [`scripts/check_mol_checkpoint.py`](scripts/check_mol_checkpoint.py) verifies one on its own and reports how many state entries it loaded.

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

Launch the fine-tuning as follows:
```bash
CHECKPOINT_PATH="${MOL_ROOT_DIR}/checkpoint/mol-1-base.pt" bash finetune_demo.sh \
  --max_steps 6 \
  --checkpoint_interval 3
```

[`finetune_demo.sh`](finetune_demo.sh) starts from a checkpoint: weights load
strictly, the optimizer and scheduler start fresh, and EMA initializes from the
loaded weights. That is deliberately distinct from resuming a run together with
its optimizer state. [docs/finetune.md](docs/finetune.md) walks through each
step, the recipe's settings and predicting with the result.


## Environment diagnosis

`mol doctor` reports the PyTorch build, the GPUs it sees, which kernels `auto`
picks, the preprocessing binaries and the data it finds. It is read-only and
loads no model; run it before filing a bug.
[docs/installation.md](docs/installation.md#verify) shows a report and what to
look for in it.

## Memory

Peak device memory grows close to quadratically with token count and depends
on the GPU. [docs/inference_memory.md](docs/inference_memory.md) has measured
points, how to fit a curve for your own card from the token count and peak
that every prediction logs, and every chunking control. When a system does not
fit, the error names the settings to turn. When it fits with room to spare,
`--infer_setting.sample_diffusion_chunk_size 5` denoises the five samples
together instead of one at a time, which is faster.

**Large inputs.** The defaults fit inputs of up to about 1,500 tokens on a
95 GiB GPU. Larger ones, such as the biggest systems in FoldBench, need the
[large-input settings](docs/inference_memory.md#large-inputs): smaller chunks,
the PyTorch triangle multiplication and expandable allocator segments. To
predict all of FoldBench, run its inputs above 1,500 tokens with them:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mol pred -i input.json -o ./output --trimul_kernel torch \
  --infer_setting.chunk_size 256 --model.msa_module.msa_chunk_size 512 \
  --infer_setting.msa_pair_transition_chunk_size 256 \
  --infer_setting.trunk_pairmixer_transition_chunk_size 32 \
  --infer_setting.trunk_pairformer_transition_chunk_size 32 \
  --infer_setting.confidence_pair_transition_chunk_size 32 \
  --infer_setting.diffusion_conditioning_row_chunk_size 384
```

On a 95 GiB GPU this completes the two largest FoldBench inputs, of 2,456 and
2,532 tokens, in about 25 minutes per seed.

## Benchmarks

**Antibody–antigen structure prediction.** (a)–(d) Top-ranked DockQ success rates with a five-seed budget. (e) Mean DockQ versus seed count K; shading denotes the 5th–95th percentiles over seed subsets. (f) Success rates across all 201,376 five-seed subsets of 32 seeds on FoldBench-AB.

<img src="assets/images/abag-overview.png" style="width: 90%; height: auto;" alt="Mol-1 overview">

**Protein–ligand and molecular glue prediction.** (a) Runs N’ Poses success rates bySuCOS-pocket similarity to training complexes; parentheses give system counts, and open diamonds denote pocket-conditioned Mol-1-constraint. (b) MGBench success rates by homology group. Success in a requires ligand RMSD < 2 Å and lDDT-PLI > 0.8, while success in b requires ligand RMSD < 2 Å, lDDT-PLI > 0.8 and protein-protein interface DockQ > 0.23. (c) Individual PoseBusters check pass rates over 64,800 predictions per method.

<img src="assets/images/pl_overview_posebusters_300dpi.png" style="width: 90%; height: auto;" alt="Mol-1 overview">


**Protein–peptide complex prediction on MolBench-Peptide-26, a 2026 Q1–Q3 bench-
mark comprising 172 interface clusters from 150 PDB entries.** (a)–(c) Success rates at DockQ > 0.23, > 0.49, and > 0.80, respectively. Mol-1 and Mol-1-constraint use 10 recycling cycles and are shown in solid and hatched blue; baselines are gray. Y-axes are truncated and use different ranges across panels.

<img src="assets/images/pp_overview.png" style="width: 90%; height: auto;" alt="Mol-1 overview">


## Documentation

- [Installation](docs/installation.md)
- [Fine-tuning on your own structures](docs/finetune.md)
- [Preparing training data](docs/prepare_training_data.md)
- [Input JSON format](docs/infer_json_format.md)
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
