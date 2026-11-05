"""PyTorch example: ResNet-18 on CIFAR-10 with full resume (model, optimizer, scheduler, AMP
scaler, RNG, position in epoch).

    pip install torch torchvision
    gpuo submit train.py --dir examples/torch_cifar --min-vram 8 --gpu-hours 6
"""

import torch
import torchvision
import torchvision.transforms as T
from torch import nn

import gpuo

EPOCHS = 30
BATCH = 128

run = gpuo.init()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)

model = torchvision.models.resnet18(num_classes=10).to(dev)
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.amp.GradScaler(enabled=dev == "cuda")
loss_fn = nn.CrossEntropyLoss()

state = run.restore(model=model, optimizer=opt, scheduler=sched, scaler=scaler)

for epoch in range(state.epoch, EPOCHS):
    # seeded per epoch so the shuffle order is the same after a resume
    g = torch.Generator().manual_seed(epoch)
    loader = torch.utils.data.DataLoader(
        data, batch_size=BATCH, shuffle=True, generator=g, num_workers=2, drop_last=True
    )
    skip = state.step_in_epoch if epoch == state.epoch else 0
    model.train()
    for step, (x, y) in enumerate(loader):
        if step < skip:
            continue  # fast-forward to where the last segment stopped
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(dev, enabled=dev == "cuda"):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        run.log(loss=loss.item(), lr=sched.get_last_lr()[0], epoch=epoch)
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
    sched.step()
    run.checkpoint(
        epoch=epoch + 1, step_in_epoch=0, model=model, optimizer=opt, scheduler=sched, scaler=scaler
    )

torch.save(model.state_dict(), "model_final.pt")
run.finish()
