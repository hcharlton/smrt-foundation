# Experiment 28: Controlled baseline

Identical to experiment 20 (supervised from scratch) but with `ds_limit=20000000` to match experiment 27's data budget. Controls for data budget differences when comparing fine-tuning (exp 27, 79%) against training from scratch (exp 20, 82%).

If this hits 82%: the 3pp gap in exp 27 was from the two-stage optimizer schedule, not from pretraining hurting.
If this drops to ~79%: the data budget itself explains the gap.
