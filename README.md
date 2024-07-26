# STIVE


## Example
* Extract concepts: `python scripts/extract_concepts.py -f data/concepts/ferrari/video_prompts.csv -o data/concepts/ferrari/concepts.json`
* Finetune concepts: `accelerate launch run/finetune_concepts.py --config configs/concepts/ferrari.yaml`
* Finetune lora: `accelerate launch run/finetune_stive.py --config configs/stive/ferrari.yaml`


