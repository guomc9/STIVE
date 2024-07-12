# Preprocess

## Extract Abstracts

python scripts/extract_abstracts.py -f data/few-shot/ski-lift-time-lapse/video_prompts.csv -o data/few-shot/ski-lift-time-lapse/abstracts.json
python scripts/extract_abstracts.py -f data/few-shot/tesla/video_prompts.csv -o data/few-shot/tesla/abstracts.json
python scripts/extract_abstracts.py -f data/few-shot/ferrari/video_prompts.csv -o data/few-shot/ferrari/abstracts.json

## Extract Correspondences

python scripts/extract_correspondences.py -d data/few-shot/ski-lift-time-lapse -t 0.018
python scripts/extract_correspondences.py -d data/few-shot/tesla -t 0.1
python scripts/extract_correspondences.py -d data/few-shot/ferrari -t 0.1


