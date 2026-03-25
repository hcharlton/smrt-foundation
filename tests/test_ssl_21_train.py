"""
Static analysis tests for ssl_21_pretrain/train.py.

Verifies that the linear probe evaluation does not run only on the main
process, which would desynchronize RNG state across ranks and cause
NCCL timeouts at the next epoch boundary.
"""

import ast
import os
import pytest


TRAIN_SCRIPT = os.path.join(
    os.path.dirname(__file__), '..', 'scripts', 'experiments',
    'ssl_21_pretrain', 'train.py'
)


@pytest.fixture(scope="module")
def train_source():
    with open(TRAIN_SCRIPT) as f:
        return f.read()


@pytest.fixture(scope="module")
def train_ast(train_source):
    return ast.parse(train_source)


class TestLinearProbeNotRankGated:
    """Ensure linear_probe_eval is NOT guarded by is_main_process."""

    def test_probe_call_not_inside_is_main_process_guard(self, train_source):
        """The linear_probe_eval() call must not be inside an
        `if ... accelerator.is_main_process` block. Running the probe
        on only one rank desynchronizes RNG state and causes NCCL timeouts."""

        # Find all lines with linear_probe_eval
        lines = train_source.splitlines()
        probe_call_lines = [
            (i, line) for i, line in enumerate(lines)
            if 'linear_probe_eval(' in line
            and not line.lstrip().startswith('#')
            and not line.lstrip().startswith('def ')
        ]
        assert probe_call_lines, "linear_probe_eval call not found in train.py"

        for line_idx, probe_line in probe_call_lines:
            # Walk backwards to find the nearest `if` guard
            indent_of_probe = len(probe_line) - len(probe_line.lstrip())
            for check_idx in range(line_idx - 1, -1, -1):
                check_line = lines[check_idx]
                stripped = check_line.lstrip()
                if not stripped or stripped.startswith('#'):
                    continue
                check_indent = len(check_line) - len(stripped)
                if check_indent < indent_of_probe and stripped.startswith('if '):
                    assert 'is_main_process' not in stripped, (
                        f"linear_probe_eval (line {line_idx + 1}) is guarded by "
                        f"is_main_process (line {check_idx + 1}). This will "
                        f"desynchronize ranks and cause NCCL timeouts."
                    )
                    break  # found the enclosing if, and it's fine


class TestBarrierAfterProbe:
    """Ensure wait_for_everyone() is called between probe and next epoch."""

    def test_wait_for_everyone_after_probe(self, train_source):
        """There must be a wait_for_everyone() call between the linear probe
        block and model.train() to synchronize all ranks before the next epoch."""

        lines = train_source.splitlines()

        # Find linear_probe_eval call
        probe_idx = None
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if ('linear_probe_eval(' in stripped
                    and not stripped.startswith('#')
                    and not stripped.startswith('def ')):
                probe_idx = i
                break
        assert probe_idx is not None, "linear_probe_eval call not found"

        # Find next model.train() after probe
        train_idx = None
        for i in range(probe_idx + 1, len(lines)):
            if 'model.train()' in lines[i]:
                train_idx = i
                break
        assert train_idx is not None, "model.train() not found after probe"

        # Check for wait_for_everyone between probe and model.train()
        barrier_found = False
        for i in range(probe_idx + 1, train_idx):
            if 'wait_for_everyone' in lines[i]:
                barrier_found = True
                break

        assert barrier_found, (
            f"No wait_for_everyone() found between linear_probe_eval "
            f"(line {probe_idx + 1}) and model.train() (line {train_idx + 1}). "
            f"A barrier is needed to synchronize ranks before the next epoch."
        )
