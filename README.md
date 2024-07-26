# STIVE


## Example
* Extract concepts: 
    - `python scripts/extract_concepts.py -f data/concepts/ferrari/video_prompts.csv -o data/concepts/ferrari/concepts.json`
* Finetune concepts: 
    - `accelerate launch run/finetune_concepts.py --config configs/concepts/ferrari.yaml`
* Inference concepts:
    - `accelerate launch run/inference_concepts.py --config configs/concepts/ferrari.yaml`
* Finetune stive: 
    - `accelerate launch run/finetune_stive.py --config configs/stive/ferrari.yaml`
* Inference stive:
    - `accelerate launch run/inference_stive.py --config configs/stive/ferrari.yaml`
* Inference stive_ptp:
    - `accelerate launch run/inference_stive_ptp.py --config configs/ptp/car-turn/ferrari.yaml`


