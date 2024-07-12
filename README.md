# STIVE

Implementation of Stable Textual Inversion Video Editing.


## Train STIVE

### One-Shot
```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch train_abstracts_clip.py --config configs/abstracts_clip/one-shot/aircraft-landing/train.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch train_stive.py --config configs/stive/one-shot/aircraft-landing/train.yaml
```
```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch train_abstracts_clip.py --config configs/abstracts_clip/one-shot/american-flag/train.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch train_stive.py --config configs/stive/one-shot/american-flag/train.yaml
```
```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch train_abstracts_clip.py --config configs/abstracts_clip/one-shot/cybertrunk/train.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch train_stive.py --config configs/stive/one-shot/cybertrunk/train.yaml
```


### Few-Shot
```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch train_abstracts_clip.py --config configs/abstracts_clip/few-shot/ski-lift-time-lapse/train.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch train_stive.py --config configs/stive/few-shot/ski-lift-time-lapse/train.yaml
```
```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch train_abstracts_clip.py --config configs/abstracts_clip/few-shot/tesla/train.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch train_stive.py --config configs/stive/few-shot/tesla/train.yaml
```
```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch train_abstracts_clip.py --config configs/abstracts_clip/few-shot/car/train.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch train_stive.py --config configs/stive/few-shot/car/train.yaml
```
```shell
CUDA_VISIBLE_DEVICES=0 accelerate launch train_abstracts_clip.py --config configs/abstracts_clip/few-shot/ferrari/train.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch train_stive.py --config configs/stive/few-shot/ferrari/train.yaml
```
