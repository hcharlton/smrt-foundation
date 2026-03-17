---
name: use-venv-for-python
description: Use the .venv virtual environment in the project root for running Python and pytest
type: feedback
---

Always activate `.venv` before running Python commands in this project: `source /Users/charlton/projects/smrt-foundation/.venv/bin/activate`

**Why:** The user has a `.venv` set up with pytest, pysam, and other required packages. System Python and conda base don't have them.

**How to apply:** Any time you need to run `python`, `pytest`, or `pip` in this project, prefix with the venv activation.
