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