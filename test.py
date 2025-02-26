import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--batchsize', type=int, default=32)
args = parser.parse_args()

import torch, torchvision
from torchvision.transforms import v2
from torcheval import metrics
from time import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################### DATA ###########################

transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, True),
])

num_classes = 10
ts = torchvision.datasets.CIFAR10('/data/toys', False, transform=transform)
ts = torch.utils.data.DataLoader(ts, args.batchsize, num_workers=4, pin_memory=True)

############################### MODEL ###########################

def LayerNorm(k): return torch.nn.GroupNorm(1, k)
model = torch.load(args.model, map_location=device, weights_only=False)

############################## METRICS ##########################

acc = metrics.MulticlassAccuracy(device=device)

############################### TEST ###########################

model.eval()
for imgs, labels in ts:
    imgs = imgs.to(device)
    labels = labels.to(device)
    preds = model(imgs)
    acc.update(preds, labels)

print(args.model, 'accuracy:', acc.compute().item())
