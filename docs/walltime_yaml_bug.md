# Silent 37.5-day SLURM TimeLimit from unquoted YAML walltime

## Summary

A config value written as `walltime: 15:00:00` — intended to request 15 hours from SLURM — was silently interpreted as a 37-day 12-hour wall limit on Gefion. The cause is a chain of two reasonable defaults (PyYAML uses YAML 1.1, which parses colon-separated integers as base 60; SLURM interprets bare-integer `--time=N` as minutes) composing into a 60× walltime inflation. The bug was present on GenomeDK too but never surfaced there, because that cluster has no equivalent of Gefion's `AssocGrpGRESMinutes` quota and because jobs always terminated at end-of-training before the inflated limit could fire.

## How it surfaced

Jobs in `scripts/experiments/ssl_53_simclr_grid_r2_step/` were running past the 15-hour walltime I thought I had requested. `scontrol show job` on one of the running jobs revealed the actual wall limit:

```text
$ scontrol show job 432467
JobId=432467 JobName=exp_size_d256_L8
   UserId=chache(1473300690) GroupId=chache(1473300690) MCS_label=cu_0030
   Priority=3612 Nice=0 Account=cu_0030 QOS=cu_0030 WCKey=*
   JobState=RUNNING Reason=None Dependency=(null)
   ...
   RunTime=16:10:37 TimeLimit=37-12:00:00 TimeMin=N/A
   SubmitTime=2026-04-23T15:57:41 EligibleTime=2026-04-23T15:57:41
   AccrueTime=2026-04-23T15:57:41
   StartTime=2026-04-23T15:57:42 EndTime=2026-05-31T03:57:42 Deadline=N/A
   ...
   ReqTRES=cpu=64,mem=256G,node=1,billing=64,gres/gpu=8
   AllocTRES=cpu=64,mem=256G,node=1,billing=64,gres/gpu=8
```

`TimeLimit=37-12:00:00` — 37 days, 12 hours. The config said:

```yaml
# scripts/experiments/ssl_53_simclr_grid_r2_step/size_d256_L8/config.yaml
resources:
  cores: 64
  memory: 256gb
  walltime: 15:00:00
  gres: gpu:8
  num_processes: 8
```

So something between the YAML file and `sbatch --time=…` inflated 15 hours into 37.5 days — a factor of exactly 60.

## The four-step chain

### 1. PyYAML parses unquoted `15:00:00` as a base-60 integer

YAML 1.1 defines a sexagesimal integer literal: a colon-separated sequence of integer digits is read as base 60. PyYAML uses YAML 1.1 by default, even through `yaml.safe_load`. Reproduced directly against the live config:

```text
$ python3 -c "
import yaml
with open('scripts/experiments/ssl_53_simclr_grid_r2_step/size_d256_L8/config.yaml') as f:
    c = yaml.safe_load(f)
wt = c['resources']['walltime']
print('walltime value:', repr(wt))
print('type:', type(wt).__name__)
"
walltime value: 54000
type: int
```

The arithmetic:

```
15 * 60^2 + 0 * 60^1 + 0 * 60^0 = 15 * 3600 + 0 + 0 = 54000
```

After `yaml.safe_load`, `walltime` is no longer a string — it is the int `54000`.

### 2. The bash wrapper passes the int through as a decimal string

`run.sh`'s `read_resource` helper (pre-fix form) was:

```bash
read_resource() {
    python3 -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
r = c.get('resources', {})
print(r.get('$1', '$2'))
"
}
```

`print(54000)` emits the string `"54000"` (no colons, no formatting). Captured in bash:

```bash
WALLTIME=$(read_resource walltime 24:00:00)
# $WALLTIME == "54000"
```

### 3. `sbatch --time=54000` is 54000 minutes, not 54000 seconds

From the SLURM docs: *"Acceptable time formats include `minutes`, `minutes:seconds`, `hours:minutes:seconds`, `days-hours`, `days-hours:minutes`, `days-hours:minutes:seconds`."* A bare integer is minutes. The submission line:

```bash
sbatch --time="$WALLTIME" ...
```

With `$WALLTIME == "54000"`, SLURM receives `--time=54000` and records it as 54000 minutes.

### 4. 54000 minutes resolves to 37 days 12 hours

```
54000 min / 60 = 900 h
900 h / 24 = 37.5 d = 37 d 12 h
```

This matches the observed `TimeLimit=37-12:00:00` exactly. Job queued at 2026-04-23 15:57:42 plus 37 d 12 h gives `EndTime=2026-05-31T03:57:42`, which the `scontrol` output confirms.

## Why it didn't surface on GenomeDK

The same YAML parser and the same `run.sh` ran on GenomeDK for months without visible symptoms. Two conditions had to line up on Gefion to make the bug observable:

1. **`AssocGrpGRESMinutes` quota.** Gefion charges *requested* time × GPUs at queue time against a per-association cap. With 37.5 d × 8 GPUs ≈ 7200 GPU-days per queued job, even one submission exhausted the GPU-hour budget I thought I was working within. GenomeDK has no equivalent per-association GPU-minute cap, so the inflated walltime was free — the jobs just sat with a wildly over-stated `TimeLimit` that nothing ever charged against.
2. **Jobs terminate on training completion, not at `TimeLimit`.** A bloated wall limit is invisible if jobs always exit before it fires. The SSL runs finish in ~15 h; neither cluster ever reached the 37.5-day wall before termination, so the oversized ceiling was never felt as a runtime effect.

Gefion did not introduce the bug. It introduced the tripwire that finally tripped it.

## Why it's hard to catch

- **The config looks correct.** `walltime: 15:00:00` is the natural human-readable way to write "15 hours." Nothing about it visually suggests base-60 weirdness.
- **YAML 1.1 sexagesimal is an obscure default.** YAML 1.2 (2009) dropped sexagesimal literals. PyYAML still defaults to 1.1 through `safe_load`. Most users do not know their parser has a base-60 mode.
- **Nothing errors anywhere in the pipeline.** `yaml.safe_load` returns a valid int. `sbatch --time=54000` accepts the flag without complaint. The job runs. There is no `RuntimeError`, rejected submission, or malformed-input warning to investigate.
- **Two systems conspire in a way that nearly cancels.** YAML compresses `15:00:00` by 60× (15 hours → 54000 "base-60 units"). SLURM then expands a bare integer by 60× (minutes → hours). Either convention in isolation is defensible; only together do they produce the 60× walltime error.
- **The existing echo in `run.sh` does not make the problem loud.** The wrapper prints `time=$WALLTIME` before submitting. Pre-fix, that printed `time=54000` — plausible-looking as "some SLURM internal," easy to skim past without pattern-matching it as "wait, that should have colons in it."
- **The only loud evidence lives in `scontrol`.** The config file, the echo line, and the `sbatch` exit code all look fine. The only visible proof of the inflated limit is `scontrol show job`'s `TimeLimit=37-12:00:00` — which you only query if you are already suspicious.

## The fix

Two changes, applied to `scripts/experiments/ssl_53_simclr_grid_r2_step/` and `run.sh`.

### Config side: quote the walltime value

All four grid configs changed:

```yaml
# Before
walltime: 15:00:00
# After
walltime: "15:00:00"
```

Quotes force PyYAML to treat the value as a string; sexagesimal parsing is skipped.

### `run.sh` side: defuse the YAML text before parsing

`read_resource` now pre-quotes any unquoted `walltime:` line via a text regex before handing the YAML body to `yaml.safe_load`. Configs that forget the quotes can no longer silently corrupt the walltime:

```bash
# Read a resource value from config.yaml, with a default fallback.
# Auto-quotes `walltime:` values so PyYAML (YAML 1.1) can't read an unquoted
# HH:MM:SS as a base-60 int (e.g. 15:00:00 → 54000, which sbatch then reads
# as 54000 minutes = 37.5 days).
read_resource() {
    python3 - "$CONFIG" "$1" "$2" <<'PYEOF'
import re, sys, yaml
config_path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
with open(config_path) as f:
    text = f.read()
text = re.sub(
    r'(?m)^(\s*walltime:[ \t]+)(?!["\'])([^\s#]+)',
    r'\1"\2"',
    text,
)
c = yaml.safe_load(text)
r = c.get('resources', {})
print(r.get(key, default))
PYEOF
}
```

The regex matches any line whose `walltime:` value is not already quoted (negative lookahead on `"` and `'`) and wraps the value in double quotes. Already-quoted values are untouched. Other fields are unaffected — only `walltime` has the sexagesimal footgun, because `memory` has unit suffixes (`256gb`), `gres` has non-numeric prefixes (`gpu:8`), and `cores`/`num_processes` are bare integers that are already in the form SLURM expects.

## Reproduction

```python
import yaml

# Minimal reproduction of the broken config
text_broken = """
resources:
  walltime: 15:00:00
"""
parsed = yaml.safe_load(text_broken)
print(repr(parsed["resources"]["walltime"]))
print(type(parsed["resources"]["walltime"]).__name__)
# -> 54000
# -> int

# Quoting defuses it
text_fixed = """
resources:
  walltime: "15:00:00"
"""
print(repr(yaml.safe_load(text_fixed)["resources"]["walltime"]))
# -> '15:00:00'
```

Feeding `54000` to `sbatch --time=`:

```
54000 min = 900 h = 37 d 12 h  →  TimeLimit=37-12:00:00
```

## Verification after the fix

The hardened `read_resource` was tested against both an unquoted input (simulating a forgotten quote) and the actual patched config:

```text
$ python3 - <tmp_unquoted.yaml> walltime 24:00:00 <<'PYEOF'
  (read_resource body as above)
PYEOF
walltime => '15:00:00'
--- rewritten YAML ---
resources:
  cores: 64
  memory: 256gb
  walltime: "15:00:00"
  gres: gpu:8
  num_processes: 8

$ python3 - scripts/experiments/ssl_53_simclr_grid_r2_step/size_d256_L8/config.yaml walltime 24:00:00 <<'PYEOF'
  (same body)
PYEOF
real config walltime => '15:00:00'
```

Both paths return the string `'15:00:00'`. An already-quoted value is not re-quoted.

## Impact and remediation

- Four jobs (grid variants `d128_L4`, `d256_L8`, `d512_L8`, `d768_L8`) were already submitted with the 37.5-day `TimeLimit` at the time of discovery. They are harmless individually — each one will still exit when training completes — but each accrues roughly 7200 GPU-days against `AssocGrpGRESMinutes` at queue time, which is the reason a quota increase had looked necessary.
- The config quoting and the `run.sh` hardening only affect *future* submissions. In-flight jobs can be brought in line with `scontrol update jobid=<id> TimeLimit=15:00:00` on the login node if desired.
- The real workload, submitted with a correct 15-hour wall limit, fits comfortably inside the default `AssocGrpGRESMinutes` cap. The pending quota-increase request to DCAI was consequently withdrawn.

## Takeaway

This is not the kind of bug one catches by being more careful with configs. It is an emergent property of composing two independently reasonable defaults — PyYAML's YAML 1.1 sexagesimal literals and SLURM's bare-integer-means-minutes convention — across a passthrough wrapper that has no context to sanity-check the string form. The right defense is a guardrail at the layer that owns the transition (the `run.sh` wrapper), not a discipline of "remember to quote time values." The fix encodes that guardrail.
