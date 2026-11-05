# gpuo

Run one training script across **your own** GPU targets — your Kaggle account, your local GPU, and (later) paid GPUs you add — with automatic checkpointing and resume.

You submit a job once. gpuo places each segment on the target with the most usable GPU time, checkpoints before the session limit, and continues on the next available target until training finishes.

```
gpuo submit train.py --gpu-hours 60 --min-vram 15
```

## What it does — and what it refuses to do

| Does | Does not |
| --- | --- |
| Schedules your job on the target you own with the most usable time left | Pool several accounts belonging to one person (Kaggle and Colab both forbid this) |
| Checkpoints model, optimizer, scheduler, AMP scaler, RNG and data position | Automate Google Colab (no API; Colab treats automation as abuse) |
| Resumes after a session cap, a crash or a disconnect | Split one model across separate notebook sessions as a "cluster" |
| Tracks estimated weekly quota per target | Store your Kaggle token (the official Kaggle CLI keeps it) |

**One-owner rule:** a job runs only on targets owned by whoever submitted it, and an owner can register only one Kaggle target. Both rules are enforced in code.

## Quick start (local)

```bash
pip install "gpuo[server] @ git+https://github.com/<you>/gpuo"
gpuo init --owner yourname
gpuo target add local --gpu RTX3060 --vram 12      # or --gpu cpu
gpuo submit train.py --dir examples/toy
gpuo serve                                          # dashboard: http://127.0.0.1:8787
```

## Add your Kaggle account

1. `pip install "gpuo[kaggle,s3]"` and set up the Kaggle CLI token as Kaggle's docs describe (`kaggle kernels list --mine` should work).
2. Kaggle kernels can't reach your laptop, so gpuo needs object storage both sides can reach. Create a Cloudflare R2 (or S3) bucket **used only by gpuo**, with an API token limited to that bucket:
   ```bash
   export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
   export GPUO_S3_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
   gpuo init --storage s3://my-gpuo-bucket/gpuo
   ```
3. In Kaggle, open any notebook → **Add-ons → Secrets**, and add `GPUO_S3_KEY_ID`, `GPUO_S3_SECRET`, `GPUO_S3_ENDPOINT` with the same values. gpuo's kernels read them there; they are never written into kernel code.
4. Register the target and check its quota:
   ```bash
   gpuo target add kaggle --username <kaggle-username> --pip-spec git+https://github.com/<you>/gpuo
   gpuo target sync kaggle --remaining 30     # type what Kaggle's settings page shows
   ```

Kernels gpuo creates are always private. Datasets should be a private Kaggle Dataset (`--kaggle-dataset you/my-data`), not part of the code bundle.

## Make your script resumable

```python
import gpuo

run = gpuo.init()
state = run.restore(model=model, optimizer=opt, scheduler=sched, scaler=scaler)

for epoch in range(state.epoch, EPOCHS):
    skip = state.step_in_epoch if epoch == state.epoch else 0
    for step, batch in enumerate(loader):  # use a per-epoch seeded shuffle
        if step < skip:
            continue
        loss = train_step(batch)
        run.log(loss=loss)
        if run.should_checkpoint() or run.should_stop():
            run.checkpoint(
                epoch=epoch,
                step_in_epoch=step + 1,
                model=model,
                optimizer=opt,
                scheduler=sched,
                scaler=scaler,
            )
            if run.should_stop():
                run.exit_for_resume()
run.finish()
```

The same script runs with plain `python train.py` (no deadline, state in `./.gpuo-standalone`). Full examples: [`examples/toy`](examples/toy/train.py) (no dependencies) and [`examples/torch_cifar`](examples/torch_cifar/train.py).

## How scheduling works

```
usable(t) = min(remaining_week_quota(t), session_cap(t)) - safety_margin
score(t)  = usable(t) × gpu_speed(t)  (+10% if the job last ran there)  - cost
```

- Remaining quota comes from gpuo's usage ledger, because Kaggle has no documented quota API. Correct it any time with `gpuo target sync`.
- Each segment gets a **lease**. A checkpoint is committed only while its lease is current, so a late "zombie" session can never overwrite newer progress.
- A segment ends as `completed`, `deadline` (checkpointed in time → next segment), `lost` (killed or disconnected → resume from last committed checkpoint) or `failed` (your code raised → job stops; gpuo does not burn quota retrying a bug).

## Commands

| Command | Purpose |
| --- | --- |
| `gpuo init` | Create `~/.gpuo` (config, SQLite DB, local storage) |
| `gpuo target add local\|kaggle\|fake` | Register a target |
| `gpuo target list` / `sync` / `disable` | Inspect and correct targets |
| `gpuo submit ENTRY --dir DIR` | Bundle the folder and queue the job |
| `gpuo status [JOB]` / `logs JOB` / `cancel JOB` | Follow jobs |
| `gpuo run` | Scheduler loop in the terminal |
| `gpuo serve` | Dashboard + scheduler |

## Development

```bash
pip install -e ".[dev]"
pytest              # includes kill-and-resume tests on a simulated target
ruff check .
```

## Status

v0.1: SDK, scheduler, local + fake + Kaggle targets, dashboard. Planned: Hugging Face `Trainer` and Lightning callbacks, MLflow, paid GPU targets (RunPod, Vast.ai via user keys), team mode. See `docs/design.md`.

License: MIT.
