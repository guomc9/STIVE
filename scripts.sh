# Train Video Diffusion Model
CUDA_VISIBLE_DEVICES=0 accelerate launch train_vid.py --config="configs/vid/loveu-tgve/train.yaml"

# Evaluate Video Diffusion Model
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_vid.py --config="configs/vid/loveu-tgve/eval.yaml"

# Train VAE
CUDA_VISIBLE_DEVICES=0 accelerate launch train_vae3d.py --config configs/vae3d/loveu-tgve/train.yaml

# Evaluate VAE
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_vae3d.py --config configs/vae3d/loveu-tgve/eval.yaml
