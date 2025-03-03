import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('method')
parser.add_argument('--num-labeled', type=int, default=1000)
parser.add_argument('--lmbda', type=float, default=1)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--labeled-batchsize', type=int, default=16)
parser.add_argument('--unlabeled-batchsize', type=int, default=112)
args = parser.parse_args()

import torchvision, torch
from torchvision.transforms import v2
from torcheval import metrics
from itertools import cycle
from time import time
import models, methods
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################################## DATA ##############################################

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, True),
    v2.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))
])
train_dataset = torchvision.datasets.CIFAR10('/data/toys', True, transforms)
val_dataset = torchvision.datasets.CIFAR10('/data/toys', False, transforms)
val_dataloader = torch.utils.data.DataLoader(val_dataset, args.labeled_batchsize+args.unlabeled_batchsize,
    pin_memory=True, num_workers=4)

# split labeled / unlabeled
SEED = 201905337
generator = torch.Generator().manual_seed(SEED)
train_labeled_dataset, train_unlabeled_dataset = torch.utils.data.random_split(train_dataset, [args.num_labeled, len(train_dataset) - args.num_labeled], generator)
train_labeled_dataloader = torch.utils.data.DataLoader(train_labeled_dataset, args.labeled_batchsize, True, num_workers=4, pin_memory=True)
train_unlabeled_dataloader = torch.utils.data.DataLoader(train_unlabeled_dataset, args.unlabeled_batchsize, True, num_workers=4, pin_memory=True)

############################################## MODEL ##############################################

model = models.WideResNet()
model.to(device)

############################################## METHODS ##############################################

weak_transform = v2.Compose([
    v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
    v2.RandomHorizontalFlip(),
])
strong_transform = v2.Compose([
    v2.RandomCrop((32, 32), padding=4, padding_mode='reflect'),
    v2.RandomHorizontalFlip(),
    v2.RandAugment(2, 10),
])

method = getattr(methods, args.method)
method = method(model, weak_transform, strong_transform)

############################################## TRAIN ##############################################

ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
optimizer = torch.optim.SGD(model.parameters(), 0.03, 0.9, weight_decay=0.0005, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000)

for epoch in range(args.epochs):
    # train semi-supervised
    tic = time()
    model.train()
    avg_sup_loss = avg_unsup_loss = 0
    semisup_n_iter = max(len(train_labeled_dataloader), len(train_unlabeled_dataloader))
    labeled_dataloader_iter = cycle(train_labeled_dataloader) if len(train_labeled_dataloader) < len(train_unlabeled_dataloader) else iter(train_labeled_dataloader)
    unlabeled_dataloader_iter = cycle(train_unlabeled_dataloader) if len(train_unlabeled_dataloader) < len(train_labeled_dataloader) else iter(train_unlabeled_dataloader)

    for (labeled, targets), (unlabeled, _) in zip(labeled_dataloader_iter, unlabeled_dataloader_iter):
        labeled, targets = next(labeled_dataloader_iter)
        unlabeled = next(unlabeled_dataloader_iter)

        labeled, targets = labeled.to(device), targets.to(device)
        unlabeled = unlabeled[0].to(device)

        supervised_loss, unsupervised_loss = method.compute_loss(labeled, targets, unlabeled)
        total_loss = supervised_loss + args.lmbda*unsupervised_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if ema_model:
            ema_model.update_parameters(model)

        avg_sup_loss += float(supervised_loss) / semisup_n_iter
        avg_unsup_loss += float(unsupervised_loss) / semisup_n_iter

    scheduler.step()
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs}', 'Train', 'Avg sup loss:', avg_sup_loss, 'Avg unsup loss:', avg_unsup_loss, 'Time:', toc-tic)

    # evaluate validation
    eval_model = ema_model
    eval_model.eval()
    acc = metrics.MulticlassAccuracy(device=device)
    for inputs, targets in val_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        acc.update(outputs, targets)
    print(f'Epoch {epoch+1}/{args.epochs}', 'Test', 'Accuracy:', acc.compute().item())

torch.optim.swa_utils.update_bn(train_labeled_dataloader, ema_model, device)