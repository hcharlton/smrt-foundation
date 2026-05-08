# TensorBoard on GenomeDK

## Where logs live

Not in the experiment directory. `_shared_train.py` writes to `training_logs/` relative to the project root (CWD at training start, set by `run.sh`'s sbatch wrap):

```
<project_root>/training_logs/<experiment_type>/<experiment_name>/
```

Always launch TB from the project root.

## One-time venv setup

TB 2.20 imports `pkg_resources`, which setuptools >= 81 dropped:

```bash
pip install 'setuptools<81'
```

## Launch (every session)

```bash
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=localhost,127.0.0.1,::1

tmux new -s tb
tensorboard --logdir training_logs --port 16006 --host 127.0.0.1
```

VS Code auto-forwards 16006. Open `http://localhost:16006`.

Filter to one run by deepening `--logdir`:

```bash
tensorboard --logdir training_logs/ssl/58_autoencoder_grid_d768_L8_h200 --port 16006 --host 127.0.0.1
```

## Gotchas

- Drop `--bind_all`. It binds to `fe-open-01.ib.gdk:<port>`, which VS Code does not auto-forward.
- Skip the default port 6006. The login node is shared and someone else usually has it.
- Without `NO_PROXY`, TB's internal gRPC data server gets 503'd through the cluster proxy. UI loads, scalars don't.
