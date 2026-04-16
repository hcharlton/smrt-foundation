"""
Plot the learning rate schedules used by experiments 37 (matched) and
38 (gradual unfreeze) as a function of training step.

Exp 37 is a single AdamW group on a cosine-with-warmup schedule over
400k steps at base lr=3e-3 (via smrt_foundation.optim.get_cosine_schedule_with_warmup,
which has a min_lr_ratio=0.05 floor).

Exp 38 is two-stage. Stage 1 (steps 1..frozen_steps): head-only at
frozen_lr on the same floor-cosine schedule. Stage 2 (remaining
steps): two param groups with separate floor-less cosine-warmup
lambdas. Encoder warms up from encoder_start_lr to encoder_lr over
encoder_warmup_steps; head warms up over pct_start * stage2_steps.
"""

import os
import sys
import math
import yaml
import argparse
import polars as pl
import altair as alt

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)


def cosine_with_floor(step, total_steps, warmup_steps, min_lr_ratio=0.05):
    """smrt_foundation.optim.get_cosine_schedule_with_warmup factor."""
    if step > total_steps:
        return min_lr_ratio
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (1.0 - min_lr_ratio) * cosine_val + min_lr_ratio


def cosine_no_floor(step, total_steps, warmup_steps, min_lr_ratio=0.0):
    """supervised_38/train.py make_cosine_warmup_lambda factor."""
    if step < warmup_steps:
        return min_lr_ratio + (1.0 - min_lr_ratio) * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def lr_exp_37(step, cfg, total_steps):
    base = float(cfg['classifier']['max_lr'])
    pct = float(cfg['classifier']['pct_start'])
    warmup = int(total_steps * pct)
    return base * cosine_with_floor(step, total_steps, warmup)


def lr_exp_38_head(step, cfg, total_steps):
    ft = cfg['finetune']
    frozen = int(ft['frozen_steps'])
    pct = float(cfg['classifier']['pct_start'])
    if step <= frozen:
        return float(ft['frozen_lr']) * cosine_with_floor(step, frozen, int(frozen * pct))
    local = step - frozen
    stage2 = total_steps - frozen
    head_warmup = int(stage2 * pct)
    return float(ft['head_lr']) * cosine_no_floor(local, stage2, head_warmup)


def lr_exp_38_encoder(step, cfg, total_steps):
    ft = cfg['finetune']
    frozen = int(ft['frozen_steps'])
    if step <= frozen:
        return None  # encoder frozen — LR undefined / not plotted
    local = step - frozen
    stage2 = total_steps - frozen
    encoder_warmup = int(ft['encoder_warmup_steps'])
    start_ratio = float(ft['encoder_start_lr']) / float(ft['encoder_lr'])
    return float(ft['encoder_lr']) * cosine_no_floor(
        local, stage2, encoder_warmup, min_lr_ratio=start_ratio
    )


def main(output_path):
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(config['exp_37_config'], 'r') as f:
        cfg37 = yaml.safe_load(f)
    with open(config['exp_38_config'], 'r') as f:
        cfg38 = yaml.safe_load(f)

    total_steps = int(config['total_steps'])
    interval = int(config['step_interval'])
    floor = float(config['min_lr_plot_floor'])
    steps = list(range(1, total_steps + 1, interval))

    curves = [
        ('37: matched', lambda s: lr_exp_37(s, cfg37, total_steps)),
        ('38: gradual (head)', lambda s: lr_exp_38_head(s, cfg38, total_steps)),
        ('38: gradual (encoder)', lambda s: lr_exp_38_encoder(s, cfg38, total_steps)),
    ]

    rows = []
    for label, fn in curves:
        for s in steps:
            lr = fn(s)
            if lr is None:
                continue
            rows.append({'step': s, 'lr': max(lr, floor), 'schedule': label})

    df = pl.DataFrame(rows)

    chart = alt.Chart(df).mark_line().encode(
        alt.X('step:Q').title(config.get('x_label', 'Step')),
        alt.Y('lr:Q', scale=alt.Scale(type='log')).title(config.get('y_label', 'Learning rate')),
        alt.Color('schedule:N').title('Schedule'),
    ).properties(
        width=config.get('width', 800),
        height=config.get('height', 400),
        title=config.get('title', ''),
    )

    chart.save(output_path)
    print(f'Saved to {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.output_path)
