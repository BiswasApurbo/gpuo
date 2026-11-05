# gpuo design notes

## Components

- **SDK** (`gpuo/sdk.py`): imported by training code. Saves and restores model, optimizer,
  scheduler, AMP scaler, RNG states and position in the epoch.
- **Agent** (`gpuo/agent.py`): runs inside a target session. Pulls the code bundle, runs the
  entrypoint, ships logs, guarantees a `status.json` when the segment ends.
- **Storage mailbox** (`gpuo/storage.py`, `gpuo/layout.py`): the only channel between the
  control plane and agents. Outbound connections only on both sides.
- **Control plane** (`gpuo/app.py`, `gpuo/db.py`, `gpuo/scheduler.py`): SQLite state, the
  placement loop, the usage ledger.
- **Targets** (`gpuo/targets/`): adapters for `local`, `fake` (testing) and `kaggle`.
- **Dashboard** (`gpuo/server.py`, `gpuo/static/index.html`): FastAPI + a single static page.

## Storage layout

```
jobs/<job>/job.json                      entrypoint + config
jobs/<job>/bundle.tar.gz                 code
jobs/<job>/lease.json                    {token, segment}  - written by the control plane
jobs/<job>/manifest.json                 {latest, history} - the only source of truth for checkpoints
jobs/<job>/ckpt/<id>.bin                 checkpoint payloads (last 3 kept)
jobs/<job>/segments/<seg>/heartbeat.json
jobs/<job>/segments/<seg>/status.json    completed | deadline | lost | failed
jobs/<job>/segments/<seg>/metrics/*.jsonl
jobs/<job>/segments/<seg>/logs/*.log
```

## Checkpoint commit protocol

1. Serialize, hash (sha256).
2. Check the lease is still ours.
3. Upload payload; re-read and verify the hash.
4. Check the lease again.
5. Write `manifest.json`.

The lease check and the manifest write are two operations, so there is a small window in
which a segment that loses its lease could still commit. The scheduler only issues a new
lease after the old segment has ended or gone silent past the heartbeat timeout, which
makes that overlap unlikely. A later version should use conditional writes (S3/R2 `If-Match`)
to close it fully.

## Known limits (v0.1)

- Kaggle quota is an estimate (no documented API); the user can correct it.
- The Kaggle CLI has no cancel command; a cancelled job's kernel stops at its next
  checkpoint when it sees the lease revoked, or at its session cap.
- Laptop asleep = no new segments are placed (running Kaggle segments continue).
