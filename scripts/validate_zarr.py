import pytest
import pysam
import zarr
import numpy as np
from scripts.bam_to_zarr import bam_to_zarr, _process_read

@pytest.fixture
def mock_config():
    return {
        'data': {
            'token_map': {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0},
            'kinetics_features': ['fi', 'ri']
        }
    }

def test_process_read():
    read = pysam.AlignedSegment()
    read.query_name = "test_read"
    read.query_sequence = "ACGT"
    read.query_qualities = pysam.qualitystring_to_array("!!!!")
    read.set_tags([("fi", [10, 20, 30, 40]), ("ri", [1, 2, 3, 4])])
    
    res = _process_read(read, ['seq', 'qual', 'fi', 'ri'])
    
    assert res is not None
    assert res['name'] == "test_read"
    assert res['seq_len'] == 4
    assert res['data']['seq'] == "ACGT"
    assert np.array_equal(res['data']['qual'], [0, 0, 0, 0])
    assert list(res['data']['fi']) == [10, 20, 30, 40]

def test_bam_to_zarr_fidelity(tmp_path, mock_config):
    bam_path = "data/ct22_5reads.bam"
    zarr_path = str(tmp_path / "output.zarr")
    optional_tags = ['fi', 'ri']
    
    bam_to_zarr(bam_path, zarr_path, 0, optional_tags, mock_config)
    
    z = zarr.open(zarr_path, mode='r')
    data = z['data'][:]
    indptr = z['indptr'][:]
    tag_to_idx = z.attrs['tag_to_idx']
    
    bam = pysam.AlignmentFile(bam_path, "rb", check_sq=False)
    valid_reads = [r for r in bam if _process_read(r, list(tag_to_idx.keys())) is not None]
    
    assert len(indptr) - 1 == len(valid_reads)
    
    lookup = {v: k for k, v in mock_config['data']['token_map'].items()}
    
    for i, read in enumerate(valid_reads):
        start, end = indptr[i], indptr[i+1]
        read_zarr = data[start:end]
        
        seq_zarr = "".join([lookup.get(x, 'N') for x in read_zarr[:, tag_to_idx['seq']]])
        assert seq_zarr == read.query_sequence
        
        qual_zarr = read_zarr[:, tag_to_idx['qual']]
        assert np.array_equal(qual_zarr, np.array(read.query_qualities))
        
        for tag in optional_tags:
            tag_zarr = read_zarr[:, tag_to_idx[tag]]
            assert np.array_equal(tag_zarr, np.array(read.get_tag(tag)))


if __name__ == "__main__":
    test_process_read()
    test_bam_to_zarr_fidelity(tmp_path='/tmp', mock_config=mock_config())