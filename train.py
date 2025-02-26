import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('method')
parser.add_argument('--sratio', type=float, default=0.3)
parser.add_argument('--uratio', type=float, default=0.7)
parser.add_argument('--lmbda', type=float, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=32)
args = parser.parse_args()

import torch, torchvision
from torchvision.transforms import v2
from semisup import semisup
from time import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################### DATA ###########################

transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, True),
])

num_classes = 10
tr = torchvision.datasets.CIFAR10('/data/toys', transform=transform)
s_tr = torch.utils.data.Subset(tr, range(int(len(tr)*args.sratio)))
s_tr = torch.utils.data.DataLoader(s_tr, args.batchsize, True, num_workers=4, pin_memory=True)
if args.uratio > 0:
    u_tr = torch.utils.data.Subset(tr, range(int(len(tr)*args.sratio), int(len(tr)*(args.sratio+args.uratio))))
    u_tr = torch.utils.data.DataLoader(u_tr, args.batchsize, True, num_workers=4, pin_memory=True)

############################### MODEL ###########################

# replace BatchNorm with LayerNorm to avoid errors when len(batch)=1
# note: GroupNorm(1)=LayerNorm
def LayerNorm(k): return torch.nn.GroupNorm(1, k)
model = torchvision.models.resnet18(num_classes=num_classes, norm_layer=LayerNorm)
model.to(device)

############################### LOSSES ###########################

soft_augment, hard_augment = semisup.augmentations(True)
ucriterion = getattr(semisup, args.method)
ucriterion = ucriterion(soft_augment=soft_augment, hard_augment=soft_augment, teacher=model)
scriterion = torch.nn.CrossEntropyLoss()

############################### TRAIN ###########################

model.train()
opt = torch.optim.AdamW(model.parameters())
for epoch in range(args.epochs):
    tic = time()
    avg_sloss = avg_uloss = 0
    for s_imgs, s_labels in s_tr:
        s_imgs = s_imgs.to(device)
        s_labels = s_labels.to(device)
        if args.uratio > 0:
            u_imgs, _ = next(iter(u_tr))
            u_imgs = u_imgs.to(device)
        # fpass
        sloss = scriterion(model(s_imgs), s_labels)
        uloss = ucriterion(model, u_imgs) if args.uratio > 0 else torch.zeros((), requires_grad=True)
        opt.zero_grad()
        loss = sloss + args.lmbda*uloss
        loss.backward()
        opt.step()
        avg_sloss += float(sloss) / len(s_tr)
        avg_uloss += float(uloss) / len(u_tr)
    toc = time()
    print(f'* Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg sup loss: {avg_sloss} - Avg unsup loss: {avg_uloss}')

torch.save(model.cpu(), args.output)
