# FT Pipeline Cookbook

Practical recipes for the four-way-partition FT evaluation pipeline. See
`docs/methodology.md` for the design rationale; this file is for the
"I want to do X" lookup.

## Prerequisites

Before any of this works, on Gefion:

- `data/01_processed/val_sets/cpg_{pos,neg}_v2.memmap/{train,val1,val2,val3}/`
  must exist (the four-way partition regen). 
- The four supervised_53 recipe configs already point at `…/val2` for
  validation (done in the same commit as `ft_eval.py`). 
- Active virtualenv at `.venv/` so the sbatch `--wrap` source line works.

## Recipe 1 — Smoke test

Smallest possible end-to-end check before committing to the full run. One SSL
size, one FT recipe, ~2-3 hours total walltime.

```bash
# 1. Reprobe just d128_L4_long
sbatch --parsable --account=cu_0030 --gres=gpu:1 --cpus-per-task=8 \
  --mem=64gb --time=04:00:00 \
  --job-name=reprobe_smoke \
  --output=scripts/experiments/ssl_58_autoencoder_grid/size_d128_L4_long/reprobe_%j.out \
  --wrap="source .venv/bin/activate && python -m scripts.ft_eval reprobe \
    --exp_dir scripts/experiments/ssl_58_autoencoder_grid/size_d128_L4_long"

# Wait for that to complete (squeue -u $USER). Confirm probe_history_val1.csv
# lands in size_d128_L4_long/ with one row per step_*.pt.

# 2. FT just midlayer recipe (depends on the reprobe jobid above)
bash run.sh scripts/experiments/supervised_53_finetune_revamp/midlayer \
  --dependency=afterok:<REPROBE_JOBID>

# 3. After FT completes, evaluate filtered to one arch
python -m scripts.ft_eval evaluate \
  --ft_exp_dir scripts/experiments/supervised_53_finetune_revamp \
  --init_name ssl_58_best \
  --recipes midlayer \
  --out /tmp/smoke_eval.csv
```

If `/tmp/smoke_eval.csv` lands with one row and `test_top1` between 0.5 and 1.0,
the pipeline is wired correctly.

## Recipe 2 — Full run for ssl_58 + supervised_53

```bash
SSL_ROOT=scripts/experiments/ssl_58_autoencoder_grid
FT_DIR=scripts/experiments/supervised_53_finetune_revamp
INIT_NAME=ssl_58_best
SIZES="d128_L4_long d256_L8_long d512_L8_long d768_L8_long d1024_L8_long"
RECIPES="midlayer lpft_lldr decoder_init recipe_match"
SBATCH_BASE="--account=cu_0030 --gres=gpu:1 --cpus-per-task=8 --mem=64gb"

# [1/3] reprobe per SSL size (parallel)
REPROBE_JOBIDS=()
for size in $SIZES; do
  exp_dir="${SSL_ROOT}/size_${size}"
  jobid=$(sbatch --parsable $SBATCH_BASE --time=04:00:00 \
    --job-name="reprobe_${size}" \
    --output="${exp_dir}/reprobe_%j.out" \
    --wrap="source .venv/bin/activate && python -m scripts.ft_eval reprobe --exp_dir ${exp_dir}")
  echo "  reprobe ${size} -> ${jobid}"
  REPROBE_JOBIDS+=($jobid)
done
REPROBE_DEPS=$(IFS=:; echo "${REPROBE_JOBIDS[*]}")

# [2/3] FT recipes (depends on reprobe)
FT_JOBIDS=()
for recipe in $RECIPES; do
  out=$(bash run.sh "${FT_DIR}/${recipe}" --dependency=afterok:${REPROBE_DEPS})
  jobid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | awk '{print $NF}')
  echo "  ft ${recipe} -> ${jobid}"
  FT_JOBIDS+=($jobid)
done
FT_DEPS=$(IFS=:; echo "${FT_JOBIDS[*]}")

# [3/3] evaluate (depends on FT)
sbatch $SBATCH_BASE --time=02:00:00 \
  --dependency=afterok:${FT_DEPS} \
  --job-name="ft_eval_${INIT_NAME}" \
  --output="${FT_DIR}/ft_eval_%j.out" \
  --wrap="source .venv/bin/activate && python -m scripts.ft_eval evaluate --ft_exp_dir ${FT_DIR} --init_name ${INIT_NAME}"
```

Final output: `${FT_DIR}/inter_ssl_eval.csv`. Monitor: `squeue -u $USER`.

## Recipe 3 — Add a new SSL family as init source

Use case: you trained ssl_60 and want it to be an init alongside ssl_58 in the
existing supervised_53 grid.

1. **Reprobe each ssl_60 size on val1** (one sbatch per size, as in Recipe 2 step 1
   but with `${SSL_ROOT}=scripts/experiments/ssl_60_ctx1024_grid` and the appropriate
   `SIZES` list).

2. **Add an `inits:` block to each of the four supervised_53 recipe configs**
   (`midlayer/config.yaml`, `lpft_lldr/config.yaml`, `decoder_init/config.yaml`,
   `recipe_match/config.yaml`):
   ```yaml
   inits:
     random:
       checkpoint: null
     ssl_58_best:
       checkpoint: 'auto_best'
       ssl_exp_dirs:
         d128_L4: 'scripts/experiments/ssl_58_autoencoder_grid/size_d128_L4_long'
         # ... existing 5 sizes ...
     ssl_60_best:        # NEW
       checkpoint: 'auto_best'
       ssl_exp_dirs:
         d128_L4: 'scripts/experiments/ssl_60_ctx1024_grid/size_d128_L4'
         # ... ssl_60 sizes ...
   ```
   That's 4 config files × ~7 lines each. `auto_best` resolution picks up the
   new `probe_history_val1.csv` automatically.

3. **Re-submit the four FT recipes**. ds_grid_v3 sees both `ssl_58_best` and
   `ssl_60_best` in `inits:` and runs both as separate combos. Existing
   `ssl_58_best` results are untouched (their `combo_dir`s already exist with
   `results.csv` and `best_ckpt.pt`); only the new `ssl_60_best` combos run.

4. **Re-run evaluate, scoped to the new init**:
   ```bash
   python -m scripts.ft_eval evaluate \
     --ft_exp_dir scripts/experiments/supervised_53_finetune_revamp \
     --init_name ssl_60_best
   ```
   Or omit `--init_name` to evaluate every init present and compare them in
   one CSV.

## Recipe 4 — Add a new FT recipe (e.g., F5)

Use case: you want to test a new fine-tuning strategy alongside F1-F4.

1. **Add a classifier class** in `smrt_foundation/model.py` (or reuse an
   existing one if the new recipe is purely an optimizer/freezing variant).
2. **Add the treatment to `VALID_TREATMENTS`** in `scripts/ds_grid_v3.py:96`.
3. **Add a dispatch branch in `_build_model`** (lines 252-274). Pattern:
   ```python
   if treatment == 'frozen_embed':
       return DirectClassifierFrozenEmbed(**common, cnn_variant=cnn_variant)
   ```
4. **If the recipe needs a custom optimizer** (e.g., parameter freezing),
   add to `_build_optimizer` (lines 321-387) — pattern matches `lpft_lldr`.
5. **Create the recipe dir**:
   ```bash
   cp -r scripts/experiments/supervised_53_finetune_revamp/midlayer \
         scripts/experiments/supervised_53_finetune_revamp/frozen_embed
   # Edit config.yaml: change experiment_name, treatment, run_message
   ```
6. **Submit** via `bash run.sh scripts/experiments/supervised_53_finetune_revamp/frozen_embed/`.
   No reprobe needed if the SSL inits are already done.
7. **Evaluate**. Either:
   - Update `DEFAULT_RECIPES` in `scripts/ft_eval.py` (line ~52) to include
     `frozen_embed`, OR
   - Pass it explicitly:
     ```bash
     python -m scripts.ft_eval evaluate \
       --ft_exp_dir scripts/experiments/supervised_53_finetune_revamp \
       --init_name ssl_58_best \
       --recipes midlayer,lpft_lldr,decoder_init,recipe_match,frozen_embed
     ```

The footgun here: `DEFAULT_RECIPES` is a static constant. If you add a recipe
dir without updating it AND without passing `--recipes`, the new recipe is
silently skipped.

## Recipe 5 — Re-run after a failure

| Failure | What to do |
|---|---|
| One reprobe job died (OOM, timeout) | Just re-submit the same sbatch command. Reprobe is resume-safe; it skips steps already in `probe_history_val1.csv` and only processes the rest. |
| One FT recipe died mid-grid | Just `bash run.sh <recipe_dir>` again. ds_grid_v3 has internal combo dispatch — failed combos will re-run, completed ones will overwrite their results.csv (and best_ckpt.pt only if the new run beats the old one's best_val_acc, which after a clean re-run from scratch will hit fresh). |
| FT recipe finished but `best_ckpt.pt` missing | Means no eval point ever fired (the run died before `first_eval_step`). Re-submit with more walltime. |
| evaluate failed mid-iteration | Just re-run. It re-reads results.csv from scratch and re-evaluates each entry — no incremental state. |
| `auto_best` resolves to wrong checkpoint | Check that `probe_history_val1.csv` exists in the SSL exp_dir. If only `probe_history.csv` exists, the selector falls back to the old-val winner. Run `ft_eval reprobe` for that size. |

## Recipe 6 — Interpreting `inter_ssl_eval.csv`

Columns:
- `init_name`, `arch`, `size`, `recipe`, `step` — identifies the FT artifact
- `val_metric` — the val2 metric the recipe was selected on (default `val_accuracy`)
- `ckpt_path` — absolute path to the `best_ckpt.pt` that was evaluated
- `test_top1`, `test_auroc`, `test_auprc`, `test_f1`, `test_loss` — val3 metrics

Rows are sorted by `test_top1` descending. The first row prints to stdout as
the "Best Finetuned Artifact" — that's what answers "which (init, arch, size,
recipe) combination won on val3."

Things to look at when reading the table:
- **SSL lift**: compare `random` init rows vs `ssl_58_best` (and `ssl_60_best`)
  init rows at matched (arch, size). The gap is the SSL→FT lift. Per
  `feedback_ssl_lift_train_size_regime.md`, this gap only matters at small
  labeled N (10k, 100k); at 8M the inits converge.
- **Recipe winner per (arch, size)**: which recipe (F1-F4) shows up in the
  top rows. If midlayer dominates → the read-out-layer hypothesis was right;
  if lpft_lldr dominates → the optimizer-shock hypothesis; etc.
- **test_top1 vs val_metric**: large gap (>2pp) is a sign of distribution
  shift between val2 and val3 — shouldn't happen with the IID partition but
  worth a spot-check.

## Recipe 7 — Add a different supervised experiment (not just new init/recipe)

Use case: you want supervised_54 with a completely different setup (different
architectures, different train_sizes, different classifier defaults), still
fine-tuned from ssl_58.

1. Make `scripts/experiments/supervised_54_X/` with its own recipe subdirs
   (e.g., `baseline/`, `freeze_first/`).
2. Each recipe gets its own `config.yaml` + `train.py` (copy from supervised_53).
3. Make sure each config's `pos_data_val` / `neg_data_val` points at
   `…/val2`. `train` paths point at `…/train`. (`val3` paths never appear in
   any config — they live in `ft_eval evaluate` only.)
4. Submit each recipe via `bash run.sh scripts/experiments/supervised_54_X/<recipe>`.
5. `ft_eval evaluate --ft_exp_dir scripts/experiments/supervised_54_X` —
   produces `supervised_54_X/inter_ssl_eval.csv`.

To compare supervised_53 winners against supervised_54 winners: run
`evaluate` separately for each, then `cat` and sort the two CSVs.
There's no built-in cross-experiment aggregator — if you find yourself doing
this routinely, that's the signal to write one (or move to gwf).

## Common failure modes (quick reference)

- **`No step_*.pt under <exp_dir>/checkpoints`** — wrong `--exp_dir` (typo
  in the size name, or the SSL experiment didn't save checkpoints).
- **`No results.csv found under <ft_exp_dir>`** — FT hasn't run yet, or
  `--ft_exp_dir` is one level too high/low.
- **`SKIP <ckpt_path>: file not found`** in evaluate output — FT combo dir
  exists but `best_ckpt.pt` doesn't. Either FT died before first eval, or
  you're pointing at the wrong recipe (path mismatch between Module B's
  manifest and where ds_grid_v3 actually saved).
- **`Metric column 'val_accuracy' not in results.csv columns`** — the FT
  runs are from an older `ds_grid_v3.py` that didn't write `val_accuracy`.
  Check the column header; pass `--metric <correct_name>`.
- **Forward pass errors in evaluate** like state_dict shape mismatch —
  `arch_cfg` saved in `best_ckpt.pt` doesn't match the model class
  `_build_model` instantiates. Means you added a new classifier class but
  the saved ckpt is from before the class existed; can't load. Re-train.

## When to graduate from this cookbook to gwf

When you're running this pipeline more than ~once per quarter across multiple
SSL families and the bash runbook starts feeling like a maintenance burden,
that's the signal to model the dependency graph in `workflow.py`:

- Reprobe per (SSL family, size) → one gwf target per pair, output is
  `probe_history_val1.csv`.
- FT per recipe → wrapped sbatch via run.sh, output is `best_ckpt.pt` paths
  (or a sentinel like `.ft_complete` to dodge the "rolling overwrite" issue).
- Evaluate → one gwf target, output is `inter_ssl_eval.csv`.

Until then, the bash runbook is fine.
