python scripts/extract_concepts.py -f data/concepts/vehicles/video_prompts.csv -o data/concepts/vehicles/concepts.json
python scripts/cache_latents.py -d data/concepts/vehicles
CUDA_VISIBLE_DEVICES=0 accelerate launch finetune_concepts.py --config configs/concepts/vehicles.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch inference_concepts.py --config configs/concepts/vehicles.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch finetune_stive.py --config configs/stive/vehicles.yaml


python scripts/extract_concepts.py -f data/concepts/vehicles-subset/video_prompts.csv -o data/concepts/vehicles-subset/concepts.json
python scripts/cache_latents.py -d data/concepts/vehicles
CUDA_VISIBLE_DEVICES=0 accelerate launch finetune_concepts.py --config configs/concepts/vehicles-subset.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch inference_concepts.py --config configs/concepts/vehicles-subset.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch finetune_stive.py --config configs/stive/vehicles-subset.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch inference_stive.py --config configs/stive/vehicles-subset.yaml

CUDA_VISIBLE_DEVICES=0 accelerate launch inference_concepts_ptp.py --config configs/ptp/vehicles-subset.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch inference_stive_ptp.py --config configs/ptp/vehicles-subset.yaml


CUDA_VISIBLE_DEVICES=0 accelerate launch finetune_stive.py --config configs/stive/car-turn.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch inference_stive.py --config configs/stive/car-turn.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch inference_stive_ptp.py --config configs/ptp/car-turn.yaml


python scripts/extract_masks.py -v data/targets/car-turn/videos/car-turn.mp4 -s jeep

python scripts/extract_masks.py -v data/concepts/lambo/videos/lambo.mp4 -s car
python scripts/extract_masks.py -v data/concepts/cybertrunk/videos/cybertrunk.mp4 -s car
python scripts/extract_masks.py -v data/concepts/ferrari/videos/ferrari.mp4 -s car
python scripts/extract_masks.py -v data/concepts/bmw/videos/bmw.mp4 -s car

python scripts/extract_concepts.py -f data/concepts/cybertrunk/video_prompts.csv -o data/concepts/cybertrunk/concepts.json
python scripts/extract_concepts.py -f data/concepts/bmw/video_prompts.csv -o data/concepts/bmw/concepts.json
python scripts/extract_concepts.py -f data/concepts/lambo/video_prompts.csv -o data/concepts/lambo/concepts.json
python scripts/extract_concepts.py -f data/concepts/ferrari/video_prompts.csv -o data/concepts/ferrari/concepts.json




CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_concepts.py --config configs/concepts/ferrari.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_concepts.py --config configs/concepts/bmw.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_concepts.py --config configs/concepts/cybertrunk.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_concepts.py --config configs/concepts/lambo.yaml


CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_stive.py --config configs/stive/ferrari.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_stive.py --config configs/stive/bmw.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_stive.py --config configs/stive/cybertrunk.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_stive.py --config configs/stive/lambo.yaml



CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_concepts_wam.py --config configs/concepts/lambo.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_stive_wam.py --config configs/stive/lambo.yaml

CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_concepts_ptp.py --config configs/ptp/lambo.yaml



## Lambo
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/concepts/sd_lambo.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/jeep_to_lambo.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/lambo.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/lambo.yaml

## Cybertruck
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/concepts/sd_cybertruck.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/cybertruck.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/cybertruck.yaml

## Ferrari
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/concepts/sd_ferrari.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/ferrari.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/ferrari.yaml

## BMW
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_concepts.py --config configs/concepts/sd_bmw.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_sd_ptp.py --config configs/sd_ptp/bmw.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/bmw.yaml


CUDA_VISIBLE_DEVICES=0 accelerate launch runs/finetune_sd_unet.py --config configs/sd_unet/jeep_to_car.yaml
CUDA_VISIBLE_DEVICES=0 accelerate launch runs/inference_lora_sd_ptp.py --config configs/sd_ptp/car.yaml