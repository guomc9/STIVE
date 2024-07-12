# Preprocess


## One-Shot
### Extract Abstracts
python scripts/extract_abstracts.py -f data/one-shot/aircraft-landing/video_prompts.csv -o data/one-shot/aircraft-landing/abstracts.json
python scripts/extract_abstracts.py -f data/one-shot/american-flag/video_prompts.csv -o data/one-shot/american-flag/abstracts.json
python scripts/extract_abstracts.py -f data/one-shot/cybertrunk/video_prompts.csv -o data/one-shot/cybertrunk/abstracts.json

### Extract Correspondences
python scripts/extract_correspondences.py -d data/one-shot/aircraft-landing -t 0.018
python scripts/extract_correspondences.py -d data/one-shot/american-flag -t 0.1
python scripts/extract_correspondences.py -d data/one-shot/cybertrunk -t 0.08



## Few-Shot
### Extract Abstracts
python scripts/extract_abstracts.py -f data/few-shot/ski-lift-time-lapse/video_prompts.csv -o data/few-shot/ski-lift-time-lapse/abstracts.json
python scripts/extract_abstracts.py -f data/few-shot/tesla/video_prompts.csv -o data/few-shot/tesla/abstracts.json
python scripts/extract_abstracts.py -f data/few-shot/ferrari/video_prompts.csv -o data/few-shot/ferrari/abstracts.json

### Extract Correspondences
python scripts/extract_correspondences.py -d data/few-shot/ski-lift-time-lapse -t 0.018
python scripts/extract_correspondences.py -d data/few-shot/tesla -t 0.08
python scripts/extract_correspondences.py -d data/few-shot/ferrari -t 0.1
