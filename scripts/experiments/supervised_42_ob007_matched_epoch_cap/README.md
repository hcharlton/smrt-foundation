# Experiment 42: OB007 matched schedule with epoch-capped step budget

Rerun of exp 37 with epoch-capped step budgets. OB007 pretrained encoder (exp 29) with exp 41's exact training schedule. The only difference from exp 41 is encoder init.

## Why

Exp 37 used fixed 400k steps, which overtrained small datasets catastrophically. The pretrained encoder was even more vulnerable — exp 37 at n=100 showed val_loss reaching 19.5 (vs exp 33's 9.8), meaning the aggressive training actively destroyed the pretrained features.

## Submission

```bash
bash run.sh scripts/experiments/supervised_42_ob007_matched_epoch_cap
```
