"""gpuo - run one training script across your own GPU targets with automatic resume.

Typical use inside a training script::

    import gpuo
    run = gpuo.init()
    state = run.restore(model=model, optimizer=opt)
    for epoch in range(state.epoch, epochs):
        for step in range(state.step_in_epoch if epoch == state.epoch else 0, steps):
            ...
            run.log(loss=loss)
            if run.should_checkpoint():
                run.checkpoint(epoch=epoch, step_in_epoch=step + 1, model=model, optimizer=opt)
            if run.should_stop():
                run.checkpoint(epoch=epoch, step_in_epoch=step + 1, model=model, optimizer=opt)
                run.exit_for_resume()
    run.finish()
"""

from .sdk import Run, RunState, init

__all__ = ["Run", "RunState", "init"]
__version__ = "0.1.0"
