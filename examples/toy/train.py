"""Dependency-free toy training loop (linear regression with SGD + random noise).

Deterministic, so an interrupted-and-resumed run must end with exactly the same weights
as an uninterrupted one. Used by the tests and as the smallest gpuo example.

    python train.py                       # standalone
    gpuo submit train.py --dir examples/toy
"""

import json
import os
import random
import time

import gpuo

EPOCHS = int(os.environ.get("TOY_EPOCHS", 5))
STEPS = int(os.environ.get("TOY_STEPS", 200))
SLEEP = float(os.environ.get("TOY_SLEEP", 0.0))  # slow it down to exercise deadlines

run = gpuo.init()
model = {"w": 0.0, "b": 0.0}
opt = {"lr": 0.05}
random.seed(1234)
state = run.restore(model=model, optimizer=opt)
if state.resumed:
    print(f"resumed at epoch {state.epoch} step {state.step_in_epoch} (global {state.global_step})")

for epoch in range(state.epoch, EPOCHS):
    start = state.step_in_epoch if epoch == state.epoch else 0
    for step in range(start, STEPS):
        x = random.uniform(-1, 1)
        y = 3.0 * x + 0.5 + random.gauss(0, 0.1)
        pred = model["w"] * x + model["b"]
        err = pred - y
        model["w"] -= opt["lr"] * err * x
        model["b"] -= opt["lr"] * err
        run.log(loss=err * err, epoch=epoch)
        if SLEEP:
            time.sleep(SLEEP)
        if run.should_checkpoint():
            run.checkpoint(epoch=epoch, step_in_epoch=step + 1, model=model, optimizer=opt)
        if run.should_stop():
            run.checkpoint(epoch=epoch, step_in_epoch=step + 1, model=model, optimizer=opt)
            print("deadline reached, checkpointed, exiting for resume")
            run.exit_for_resume()
    opt["lr"] *= 0.9  # per-epoch LR decay: part of the state that must survive resume

result = {"w": round(model["w"], 10), "b": round(model["b"], 10)}
with open(os.environ.get("TOY_RESULT", "result.json"), "w") as f:
    json.dump(result, f)
print("final", result)
run.finish(**result)
