
python scripts/extract_concepts.py -f data/concepts/vehicles/video_prompts.csv -o data/concepts/vehicles/concepts.json
python scripts/cache_latents.py -d data/concepts/vehicles
CUDA_VISIBLE_DEVICES=0 accelerate launch finetune_concepts_clip.py --config configs/concepts_clip/vehicles.yaml