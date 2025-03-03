#!/bin/bash
METHODS="Supervised FixMatch MixMatch DINO"
for METHOD in $METHODS; do
    echo "train $METHOD"
    python train.py model-$METHOD.pth $METHOD
done
