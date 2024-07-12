# STIVE

## Train AbstractsCLIP
CUDA_VISIBLE_DEVICES=0 accelerate launch train_abstracts_clip.py --config configs/abstracts_clip/few-shot/ski-lift-time-lapse/train.yaml

## Train STIVE
CUDA_VISIBLE_DEVICES=0 accelerate launch train_stive.py --config configs/stive/few-shot/ski-lift-time-lapse/train.yaml
