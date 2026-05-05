"""
Tests for scripts/bam_to_labeled_memmap.py.

Test data: a temp BAM containing the first N=1000 reads of the real yoran BAM,
paired with a temp labels file derived from the real yoran tissue-labels file
filtered to those same reads. This exercises the actual ``<cell>/<zmw>/ccs
<tissue>`` line format, the real read-name conventions, and the real BAM tag
set (which includes sm/sx) -- not synthetic stand-ins.

The first 1000 reads of yoran span ~10 cells and all 10 tissues, so the
fixture has enough diversity for tissue-filter, cap, and exclusion tests.

Setup cost: ~1s to stream the subset BAM, ~8s to filter the 24M-line labels
file. Done once per session.

The ZMW-disambiguation test stays a pure unit test of ``parse_labels`` -- it
needs a synthetic labels string with controlled cell collisions and does not
need a BAM at all.
"""

import json
import os

import numpy as np
import polars as pl
import pysam
import pytest
import yaml

from scripts.bam_to_labeled_memmap import (
    _build_seq_lookup,
    _process_read,
    bam_to_labeled_memmap,
    parse_labels,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'data.yaml')

REAL_YORAN_BAM = os.path.join(
    os.path.dirname(__file__), '..',
    'data', '00_raw', 'unlabeled', 'yoran_kinetics_diploid.bam',
)
REAL_YORAN_LABELS = os.path.join(
    os.path.dirname(__file__), '..',
    'data', '01_processed', 'val_sets', 'yoran_read_labels.txt',
)

# Number of leading reads of the real yoran BAM to stream into the test
# subset. Empirically the first 1000 reads cover all 10 tissues across 10
# cells with at least 60 reads per tissue -- plenty for cap/exclusion tests.
N_TEST_READS = 1000

TEST_CONTEXT = 2048    # comfortably less than min HiFi read length
TEST_SHARD_SIZE = 64   # gives a few shards across ~1000 rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_bam_read_raw(bam_path, query_name, features, lookup_table):
    """
    Re-build the (L, n_features) uint8 array that bam_to_labeled_memmap would
    have produced for ``query_name`` -- by re-extracting BAM tags directly.
    Used as the trusted reference in fidelity tests.
    """
    with pysam.AlignmentFile(bam_path, 'rb', check_sq=False) as bam:
        for read in bam:
            if read.query_name != query_name:
                continue
            seq = read.query_sequence
            L = len(seq)
            out = np.zeros((L, len(features)), dtype=np.uint8)
            seq_bytes = np.frombuffer(seq.upper().encode('ascii'), dtype=np.uint8)
            for i, feat in enumerate(features):
                if feat == 'seq':
                    out[:, i] = lookup_table[seq_bytes]
                elif feat == 'mask':
                    pass
                elif feat in ('ri', 'rp'):
                    arr = np.array(read.get_tag(feat), dtype=np.uint8)
                    out[:, i] = arr[::-1].copy()
                else:
                    out[:, i] = np.array(read.get_tag(feat), dtype=np.uint8)
            return out
    raise ValueError(f"read {query_name!r} not found in {bam_path}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='session')
def config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope='session')
def seq_lookup(config):
    return _build_seq_lookup(config['data']['token_map'])


@pytest.fixture(scope='session')
def yoran_subset(tmp_path_factory):
    """
    Build a session-scoped (subset_bam, subset_labels) pair that contains
    real yoran reads + real labels for those reads.

    Returns dict:
      bam_path:   temp BAM with the first N_TEST_READS reads of the real yoran BAM
      labels_path: temp labels file filtered to those reads
      label_map:  parsed dict {read_name: tissue} from the temp labels file
      tissues:    sorted list of tissues present
      tissue_counts: {tissue: n_reads}
      n_reads:    number of reads written to the subset BAM
    """
    if not os.path.exists(REAL_YORAN_BAM):
        pytest.skip(f"yoran BAM not present: {REAL_YORAN_BAM}")
    if not os.path.exists(REAL_YORAN_LABELS):
        pytest.skip(f"yoran labels not present: {REAL_YORAN_LABELS}")

    tmpdir = tmp_path_factory.mktemp('yoran_subset')
    sub_bam = str(tmpdir / 'yoran_subset.bam')
    sub_labels = str(tmpdir / 'yoran_subset_labels.txt')

    # 1. Stream first N reads of the real yoran BAM to the subset BAM
    names = []
    with pysam.AlignmentFile(REAL_YORAN_BAM, 'rb', check_sq=False) as in_bam:
        with pysam.AlignmentFile(sub_bam, 'wb', template=in_bam) as out_bam:
            for i, r in enumerate(in_bam):
                if i >= N_TEST_READS:
                    break
                out_bam.write(r)
                names.append(r.query_name)
    name_set = set(names)

    # 2. Filter the real labels file to those reads, preserving original line
    #    formatting (including any trailing whitespace).
    with open(REAL_YORAN_LABELS, 'r') as f, open(sub_labels, 'w') as out:
        for line in f:
            parts = line.split()
            if parts and parts[0] in name_set:
                out.write(line)

    # 3. Reload labels for inspection
    label_map = {}
    with open(sub_labels, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                label_map[parts[0]] = parts[1]

    from collections import Counter
    tissue_counts = dict(Counter(label_map.values()))
    tissues = sorted(tissue_counts.keys())

    return {
        'bam_path': sub_bam,
        'labels_path': sub_labels,
        'label_map': label_map,
        'tissues': tissues,
        'tissue_counts': tissue_counts,
        'n_reads': len(names),
    }


@pytest.fixture(scope='session')
def output_dir(yoran_subset, config, tmp_path_factory):
    """Run the script once per session against the yoran subset."""
    out = str(tmp_path_factory.mktemp('labeled_memmap'))
    bam_to_labeled_memmap(
        bam_path=yoran_subset['bam_path'],
        label_path=yoran_subset['labels_path'],
        output_dir=out,
        config=config,
        tissues=yoran_subset['tissues'],
        context=TEST_CONTEXT,
        max_reads_per_tissue=0,
        optional_tags=[],
        seed=0,
        shard_size=TEST_SHARD_SIZE,
    )
    return out


# ---------------------------------------------------------------------------
# The read-attribution test (the test the user explicitly asked for)
# ---------------------------------------------------------------------------


def test_specific_read_attribution(yoran_subset, output_dir, seq_lookup):
    """
    For a known read_name (real yoran read), the manifest, the labels
    sidecar, and the shard data row must all agree -- and the shard bytes
    must match the BAM's bytes when re-extracted, sliced at the recorded
    crop_start.
    """
    manifest = pl.read_parquet(os.path.join(output_dir, 'manifest.parquet'))
    assert manifest.height > 0, "manifest is empty -- the join produced nothing"

    target_name = manifest.row(0, named=True)['read_name']
    expected_tissue = yoran_subset['label_map'][target_name]

    rows = manifest.filter(pl.col('read_name') == target_name)
    assert rows.height == 1, f"{target_name} should appear exactly once, got {rows.height}"
    row = rows.row(0, named=True)

    # 1. Tissue label correctness in the manifest.
    assert row['tissue_str'] == expected_tissue

    # 2. Tissue label correctness in the labels sidecar.
    labels = np.load(os.path.join(output_dir, f"labels_{row['shard_idx']:05d}.npy"))
    assert int(labels[row['row_idx'], 0]) == int(row['tissue_id'])
    assert int(labels[row['row_idx'], 1]) == int(row['cell_id'])

    # 3. Kinetics correctness: BAM bytes -> ri/rp aligned -> slice at crop_start
    #    must equal the shard row exactly (uint8, no rounding).
    schema = json.load(open(os.path.join(output_dir, 'schema.json')))
    features = schema['features']
    ctx = schema['context']

    bam_full = _load_bam_read_raw(yoran_subset['bam_path'], target_name, features, seq_lookup)
    expected_window = bam_full[row['crop_start']:row['crop_start'] + ctx]

    shard = np.load(os.path.join(output_dir, f"shard_{row['shard_idx']:05d}.npy"))
    written_window = shard[row['row_idx']]

    np.testing.assert_array_equal(written_window, expected_window)


def test_reverse_kinetics_alignment(yoran_subset, output_dir, seq_lookup):
    """
    Direct check that the shard's ``ri`` column at output position j equals
    the BAM's raw ri tag at index L - 1 - (crop_start + j). Catches off-by-one
    errors in the alignment direction, the class of bug from v1.
    """
    manifest = pl.read_parquet(os.path.join(output_dir, 'manifest.parquet'))
    target_name = manifest.row(0, named=True)['read_name']
    row = manifest.filter(pl.col('read_name') == target_name).row(0, named=True)

    schema = json.load(open(os.path.join(output_dir, 'schema.json')))
    features = schema['features']
    ctx = schema['context']
    ri_idx = features.index('ri')
    rp_idx = features.index('rp')

    shard = np.load(os.path.join(output_dir, f"shard_{row['shard_idx']:05d}.npy"))
    written = shard[row['row_idx']]
    crop_start = row['crop_start']

    with pysam.AlignmentFile(yoran_subset['bam_path'], 'rb', check_sq=False) as bam:
        for read in bam:
            if read.query_name != target_name:
                continue
            L = read.query_length
            ri = np.array(read.get_tag('ri'), dtype=np.uint8)
            rp = np.array(read.get_tag('rp'), dtype=np.uint8)
            # Forward-strand position j corresponds to BAM index L-1-j
            # Sample a handful of positions across the window for speed.
            for j in (0, ctx // 4, ctx // 2, 3 * ctx // 4, ctx - 1):
                fwd_pos = crop_start + j
                bam_idx = L - 1 - fwd_pos
                assert written[j, ri_idx] == ri[bam_idx], (
                    f"ri mismatch at j={j}: shard={written[j, ri_idx]}, "
                    f"bam ri[L-1-{fwd_pos}]={ri[bam_idx]}"
                )
                assert written[j, rp_idx] == rp[bam_idx]
            return
    pytest.fail(f"read {target_name} not found")


def test_real_labels_format_is_parsed(yoran_subset, output_dir):
    """
    The real yoran labels file uses ``<read_name> <tissue>\\n`` (with
    occasional trailing whitespace). Verify every read written to a shard has
    a non-empty tissue/cell string in the manifest (i.e., the parser didn't
    silently drop or mislabel anything when fed the real format).
    """
    manifest = pl.read_parquet(os.path.join(output_dir, 'manifest.parquet'))
    assert manifest.height > 0
    assert manifest['tissue_str'].null_count() == 0
    assert manifest['cell_str'].null_count() == 0
    # Every manifest tissue must appear in the source label_map for the same read
    sample = manifest.head(min(20, manifest.height))
    for row in sample.iter_rows(named=True):
        assert row['tissue_str'] == yoran_subset['label_map'][row['read_name']]


# ---------------------------------------------------------------------------
# Defensive tests
# ---------------------------------------------------------------------------


def test_zmw_disambiguation(tmp_path):
    """
    Same ZMW integer in two distinct cells with different tissues must
    produce two distinct entries in the parsed label_map and two distinct
    cell_to_id entries. Pure unit test of parse_labels -- no BAM needed,
    so it does not depend on the real yoran subset.
    """
    label_path = tmp_path / 'zmw_labels.txt'
    with open(label_path, 'w') as f:
        f.write("cellAprefix_movie/1234/ccs tissueX\n")
        f.write("cellBprefix_movie/1234/ccs tissueY\n")
        f.write("cellAprefix_movie/9999/ccs tissueX\n")

    label_map, cell_to_id = parse_labels(str(label_path), ['tissueX', 'tissueY'])

    assert label_map.get('cellAprefix_movie/1234/ccs') == 'tissueX'
    assert label_map.get('cellBprefix_movie/1234/ccs') == 'tissueY'
    assert label_map.get('cellAprefix_movie/9999/ccs') == 'tissueX'
    assert len(label_map) == 3

    assert 'cellAprefix_movie' in cell_to_id
    assert 'cellBprefix_movie' in cell_to_id
    assert cell_to_id['cellAprefix_movie'] != cell_to_id['cellBprefix_movie']


def test_excluded_tissue_is_dropped(yoran_subset, config, tmp_path_factory):
    """Restricting to a single tissue excludes all others from the manifest."""
    target_tissue = yoran_subset['tissues'][0]
    out = str(tmp_path_factory.mktemp('one_tissue'))
    bam_to_labeled_memmap(
        bam_path=yoran_subset['bam_path'],
        label_path=yoran_subset['labels_path'],
        output_dir=out,
        config=config,
        tissues=[target_tissue],
        context=TEST_CONTEXT,
        max_reads_per_tissue=0,
        seed=0,
        shard_size=TEST_SHARD_SIZE,
    )
    m = pl.read_parquet(os.path.join(out, 'manifest.parquet'))
    if m.height > 0:
        assert set(m['tissue_str'].unique().to_list()) == {target_tissue}


def test_too_short_reads_are_dropped(yoran_subset, config, tmp_path_factory):
    """A context larger than any read leaves the manifest empty."""
    out = str(tmp_path_factory.mktemp('too_short'))
    huge_context = 100_000  # bigger than any HiFi read
    bam_to_labeled_memmap(
        bam_path=yoran_subset['bam_path'],
        label_path=yoran_subset['labels_path'],
        output_dir=out,
        config=config,
        tissues=yoran_subset['tissues'],
        context=huge_context,
        max_reads_per_tissue=0,
        seed=0,
        shard_size=TEST_SHARD_SIZE,
    )
    m = pl.read_parquet(os.path.join(out, 'manifest.parquet'))
    assert m.height == 0
    shards = [f for f in os.listdir(out) if f.startswith('shard_')]
    assert len(shards) == 0


def test_max_reads_per_tissue_respected(yoran_subset, config, tmp_path_factory):
    """Per-tissue cap must hold."""
    cap = 3
    out = str(tmp_path_factory.mktemp('capped'))
    bam_to_labeled_memmap(
        bam_path=yoran_subset['bam_path'],
        label_path=yoran_subset['labels_path'],
        output_dir=out,
        config=config,
        tissues=yoran_subset['tissues'],
        context=TEST_CONTEXT,
        max_reads_per_tissue=cap,
        seed=0,
        shard_size=TEST_SHARD_SIZE,
    )
    m = pl.read_parquet(os.path.join(out, 'manifest.parquet'))
    counts = m.group_by('tissue_str').len()
    if counts.height > 0:
        assert counts['len'].max() <= cap


def test_seed_determinism(yoran_subset, config, tmp_path_factory):
    """Same seed twice -> identical manifests and bit-identical shards."""
    out_a = str(tmp_path_factory.mktemp('det_a'))
    out_b = str(tmp_path_factory.mktemp('det_b'))
    for out in (out_a, out_b):
        bam_to_labeled_memmap(
            bam_path=yoran_subset['bam_path'],
            label_path=yoran_subset['labels_path'],
            output_dir=out,
            config=config,
            tissues=yoran_subset['tissues'],
            context=TEST_CONTEXT,
            max_reads_per_tissue=0,
            seed=7,
            shard_size=TEST_SHARD_SIZE,
        )
    m_a = pl.read_parquet(os.path.join(out_a, 'manifest.parquet')).sort(['shard_idx', 'row_idx'])
    m_b = pl.read_parquet(os.path.join(out_b, 'manifest.parquet')).sort(['shard_idx', 'row_idx'])
    assert m_a.equals(m_b)

    for shard in sorted(f for f in os.listdir(out_a) if f.startswith('shard_')):
        a = np.load(os.path.join(out_a, shard))
        b = np.load(os.path.join(out_b, shard))
        np.testing.assert_array_equal(a, b)


def test_optional_tags_round_trip(yoran_subset, config, tmp_path_factory, seq_lookup):
    """
    Pass --optional_tags sm sx and verify the shard channel count, schema,
    and one row's optional-tag column matches the BAM's tag values at the
    recorded crop_start. Yoran BAM has sm/sx so this exercises the full
    optional-tag path.
    """
    out = str(tmp_path_factory.mktemp('optional_tags'))
    bam_to_labeled_memmap(
        bam_path=yoran_subset['bam_path'],
        label_path=yoran_subset['labels_path'],
        output_dir=out,
        config=config,
        tissues=yoran_subset['tissues'],
        context=TEST_CONTEXT,
        max_reads_per_tissue=0,
        optional_tags=['sm', 'sx'],
        seed=0,
        shard_size=TEST_SHARD_SIZE,
    )

    schema = json.load(open(os.path.join(out, 'schema.json')))
    expected_features = ['seq', 'fi', 'fp', 'ri', 'rp', 'sm', 'sx', 'mask']
    assert schema['features'] == expected_features
    assert schema['output_shape'] == ['N', TEST_CONTEXT, len(expected_features)]

    m = pl.read_parquet(os.path.join(out, 'manifest.parquet'))
    assert m.height > 0
    row = m.row(0, named=True)
    shard = np.load(os.path.join(out, f"shard_{row['shard_idx']:05d}.npy"))
    written = shard[row['row_idx']]
    assert written.shape == (TEST_CONTEXT, len(expected_features))

    bam_full = _load_bam_read_raw(yoran_subset['bam_path'], row['read_name'], expected_features, seq_lookup)
    expected_window = bam_full[row['crop_start']:row['crop_start'] + TEST_CONTEXT]
    np.testing.assert_array_equal(written, expected_window)
