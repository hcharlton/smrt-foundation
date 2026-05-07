"""Visually verify that ri/rp are aligned to forward-strand position j.

Run: `python -m report.probe_tissue_yoran.eda_reverse_kinetics_alignment`

The build script `scripts/bam_to_labeled_memmap.py` reverses the per-read
ri/rp arrays via `[::-1].copy()` so column j of the output corresponds to
forward-strand position j (the same column as fi/fp). The v1 archived script
had a bug that co-flipped seq+kinetics on the reverse view and silently
misaligned ri/rp against the reverse-complemented sequence (see
`docs/methodology.md:243-249`). This EDA confirms the v2 / tissue-build
alignment is correct.

Three visualisations + one summary CSV from a 5k random sample of the train
split, raw uint8 (no norm, no crop):

1. Per-position mean across reads, four kinetics channels overlaid.
2. Per-(forward-base-identity, channel) mean kinetics, with two adjacent
   panels: fi-grouped-by-seq[j] vs ri-grouped-by-seq[j], and the same with
   ri grouped by complement(seq[j]). A correctly aligned dataset has
   `fi-by-B ≈ ri-by-complement(B)` because both measure polymerase
   kinetics at `complement(B)` (forward fi at base B = polymerase reading
   B; reverse ri at forward pos j with seq[j]=B = polymerase reading the
   reverse-strand base, which is complement(B)).
3. fi vs ri kinetics scatter at j=512 and j=1536. Two panels. If alignment
   is wrong, both panels look identical (per-position pairing is
   meaningless); if correct, each j is a different aligned coordinate so
   the panels look different.

Headline summary CSV: Pearson r between fi[j] and ri[j] averaged over j vs
between fi[j] and ri[2047-j]. The first should dominate the second iff
alignment is correct.
"""

import os
import numpy as np
import polars as pl
import altair as alt

from . import _shared


SAMPLE_N = 2_000
CONTEXT = 4096
CHANNEL_NAMES = ['fi', 'fp', 'ri', 'rp']
BASE_NAMES = ['A', 'C', 'G', 'T']
BASE_TOKS = [_shared.TOK_A, _shared.TOK_C, _shared.TOK_G, _shared.TOK_T]
COMPLEMENT = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}


def _load():
    """Load 5k train samples at full ctx, raw (no norm)."""
    return _shared.load_split(
        'train', norm_fn=None, context=CONTEXT, limit=SAMPLE_N,
    )


def per_position_mean(X, out_path):
    """4-channel mean across reads at each forward-strand position."""
    kin = X[..., _shared.KIN_COLS]  # (N, T, 4)
    means = kin.mean(axis=0)  # (T, 4)

    # Subsample positions to keep the figure light: every 4th position.
    idx = np.arange(0, means.shape[0], 4)
    rows = []
    for ci, name in enumerate(CHANNEL_NAMES):
        for j in idx:
            rows.append({'position': int(j), 'channel': name, 'mean': float(means[j, ci])})
    df = pl.DataFrame(rows)

    chart = alt.Chart(df).mark_line(opacity=0.7).encode(
        alt.X('position:Q').title('Forward-strand position j'),
        alt.Y('mean:Q').title('Mean raw kinetics value (uint8)'),
        alt.Color('channel:N'),
    ).properties(
        width=700, height=280,
        title='Per-position mean kinetics across 5k train reads',
    )
    chart.save(out_path)
    print(f"  saved {out_path}")


def per_base_mean(X, out_path):
    """Per-(channel, base) mean kinetics; tests reverse-strand alignment.

    Two facets:
      - "ri grouped by seq[j]": rev kinetics binned by the FORWARD base at
        position j. Should look DIFFERENT from fi-by-seq[j] (different bases).
      - "ri grouped by complement(seq[j])": rev kinetics binned by the
        REVERSE-STRAND base at position j. Should approximately MATCH
        fi grouped by seq[j], because both measure polymerase kinetics at
        the same base identity (just on different strands).
    """
    kin = X[..., _shared.KIN_COLS]  # (N, T, 4)
    seq = X[..., _shared.SEQ_COL].astype(np.int64)  # (N, T)

    # complement table (A<->T, C<->G, N->N)
    comp_seq = _shared.COMPLEMENT_TOK[seq]

    rows = []
    for ci, ch_name in enumerate(CHANNEL_NAMES):
        ch = kin[..., ci]  # (N, T)
        for b_tok, b_name in zip(BASE_TOKS, BASE_NAMES):
            # group by forward base
            mask = seq == b_tok
            v_fwd = ch[mask].mean() if mask.any() else float('nan')
            rows.append({
                'channel': ch_name,
                'grouping': 'by forward base seq[j]',
                'base': b_name,
                'mean': float(v_fwd),
            })
            # group by reverse-strand base = complement of forward base
            mask2 = comp_seq == b_tok
            v_rev = ch[mask2].mean() if mask2.any() else float('nan')
            rows.append({
                'channel': ch_name,
                'grouping': 'by reverse base complement(seq[j])',
                'base': b_name,
                'mean': float(v_rev),
            })
    df = pl.DataFrame(rows)

    # Persist the table for offline inspection.
    df.write_csv(out_path.replace('.svg', '.csv'))

    chart = alt.Chart(df).mark_bar().encode(
        alt.X('base:N').title('Base identity at the grouping coordinate'),
        alt.Y('mean:Q').title('Mean raw kinetics (uint8)'),
        alt.Color('channel:N'),
        alt.XOffset('channel:N'),
    ).properties(width=240, height=240).facet(
        column=alt.Column('grouping:N').title(None),
        title=(
            'Per-base mean kinetics. Correct alignment: '
            'fi-by-fwd-base ≈ ri-by-rev-base (right panel matches across channels).'
        ),
    )
    chart.save(out_path)
    print(f"  saved {out_path}")


def fi_vs_ri_scatter(X, out_path, positions=(512, 1536)):
    """fi vs ri scatter at two distinct positions. Subsample to 2k dots/panel."""
    kin = X[..., _shared.KIN_COLS]
    fi_col = CHANNEL_NAMES.index('fi')
    ri_col = CHANNEL_NAMES.index('ri')

    rng = np.random.default_rng(0)
    idx = rng.choice(X.shape[0], size=min(2000, X.shape[0]), replace=False)

    rows = []
    for j in positions:
        for k in idx:
            rows.append({
                'position': f'j={j}',
                'fi': float(kin[k, j, fi_col]),
                'ri': float(kin[k, j, ri_col]),
            })
    df = pl.DataFrame(rows)

    chart = alt.Chart(df).mark_circle(size=15, opacity=0.3).encode(
        alt.X('fi:Q').title('fi (forward IPD, raw uint8)'),
        alt.Y('ri:Q').title('ri (reverse IPD aligned to fwd j, raw uint8)'),
    ).properties(width=320, height=320).facet(
        column=alt.Column('position:N'),
        title='fi vs ri at two positions. Different j should show distinct distributions.',
    )
    chart.save(out_path)
    print(f"  saved {out_path}")


def pearson_summary(X, out_path):
    """Pearson r between fi[j], ri[j] vs fi[j], ri[T-1-j].

    Mean over j of per-position correlation across reads. The aligned
    pairing should give a small but stable signal; the reversed pairing is
    a coordinate-mirrored test that should be ~zero.
    """
    kin = X[..., _shared.KIN_COLS].astype(np.float64)  # (N, T, 4)
    fi = kin[..., 0]
    ri = kin[..., 2]
    T = fi.shape[1]

    # Per-position Pearson r averaged over j. Subsample positions to keep
    # the inner loop tractable: 256 evenly spaced.
    js = np.linspace(0, T - 1, 256).astype(int)
    rs_aligned, rs_reversed = [], []
    for j in js:
        a = fi[:, j]
        b_aligned = ri[:, j]
        b_reversed = ri[:, T - 1 - j]
        if a.std() > 0 and b_aligned.std() > 0:
            rs_aligned.append(np.corrcoef(a, b_aligned)[0, 1])
        if a.std() > 0 and b_reversed.std() > 0:
            rs_reversed.append(np.corrcoef(a, b_reversed)[0, 1])

    summary = {
        'r_fi_ri_aligned_mean': float(np.mean(rs_aligned)),
        'r_fi_ri_aligned_std': float(np.std(rs_aligned)),
        'r_fi_ri_reversed_mean': float(np.mean(rs_reversed)),
        'r_fi_ri_reversed_std': float(np.std(rs_reversed)),
        'n_reads': int(X.shape[0]),
        'n_positions_sampled': int(len(js)),
    }
    pl.DataFrame([summary]).write_csv(out_path)
    print(f"  saved {out_path}")
    print("  ", summary)
    return summary


def main():
    _shared.ensure_dirs()
    print(f"Loading {SAMPLE_N} train samples at ctx={CONTEXT} (raw, no norm) ...")
    data = _load()
    X = data['X']
    print(f"  X.shape = {X.shape}")

    print("\n[1/4] Per-position mean kinetics ...")
    per_position_mean(X, os.path.join(_shared.FIGURES_DIR, 'reverse_kinetics_alignment_per_position.svg'))

    print("\n[2/4] Per-base mean kinetics (alignment test) ...")
    per_base_mean(X, os.path.join(_shared.FIGURES_DIR, 'reverse_kinetics_alignment_per_base.svg'))

    print("\n[3/4] fi vs ri scatter at j=512 and j=1536 ...")
    fi_vs_ri_scatter(X, os.path.join(_shared.FIGURES_DIR, 'reverse_kinetics_alignment_scatter.svg'))

    print("\n[4/4] Pearson r summary ...")
    pearson_summary(X, os.path.join(_shared.RESULTS_DIR, 'reverse_kinetics_alignment_summary.csv'))


if __name__ == '__main__':
    main()
