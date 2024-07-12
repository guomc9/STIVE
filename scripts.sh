python scripts/extract_abstracts.py -f data/few-shot/ski-lift-time-lapse/video_prompts.csv -o data/few-shot/ski-lift-time-lapse/abstracts.json
python scripts/extract_correspondences.py -d data/few-shot/ski-lift-time-lapse -t 0.018

python scripts/extract_abstracts.py -f data/few-shot/drift-turn/video_prompts.csv -o data/few-shot/drift-turn/abstracts.json
python scripts/extract_correspondences.py -d data/few-shot/drift-turn -t 0.1

python scripts/extract_abstracts.py -f data/few-shot/tesla/video_prompts.csv -o data/few-shot/tesla/abstracts.json
python scripts/extract_correspondences.py -d data/few-shot/tesla -t 0.1





# Train Video Diffusion Model
CUDA_VISIBLE_DEVICES=0 accelerate launch train_vid.py --config="configs/vid/loveu-tgve/train.yaml"

# Evaluate Video Diffusion Model
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_vid.py --config="configs/vid/loveu-tgve/eval.yaml"

# Train VAE
CUDA_VISIBLE_DEVICES=0 accelerate launch train_vae3d.py --config configs/vae3d/loveu-tgve/train.yaml

# Evaluate VAE
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_vae3d.py --config configs/vae3d/loveu-tgve/eval.yaml

# Train Abstract CLIP
CUDA_VISIBLE_DEVICES=0 accelerate launch train_abstract_clip.py --config configs/abstract_clip/one-shot/ski-lift-time-lapse/train.yaml

CUDA_VISIBLE_DEVICES=1 accelerate launch train_tp_tive_with_clip.py --config configs/tp_tive_with_clip/few-shot/ski-lift-time-lapse/train.yaml


# Extract abstracts from csv
python scripts/extract_abstracts.py -f data/one-shot/american-flag/one-shot.csv -o data/one-shot/american-flag/abstracts.json

python scripts/extract_abstracts.py -f data/one-shot/aircraft-landing/one-shot.csv -o data/one-shot/aircraft-landing/abstracts.json

python scripts/extract_abstracts.py -f data/one-shot/cat-in-the-sun/one-shot.csv -o data/one-shot/cat-in-the-sun/abstracts.json

python scripts/extract_abstracts.py -f data/few-shot/ski-lift-time-lapse/video_prompts.csv -o data/few-shot/ski-lift-time-lapse/abstracts.json




python scripts/extract_correspondences.py -d data/few-shot/ski-lift-time-lapse -t 0.015

python scripts/recons_score.py -g .log/vae3d/loveu-tgve/base/2024-06-30T13-44-00/samples/sample-3200/A\ cat\ in\ the\ grass\ in\ the\ sun..gif -m data/loveu-tgve/videos/cat-in-the-sun.mp4
# 33.36, 0.9045

python scripts/recons_score.py -g .log/vae3d/loveu-tgve/base/2024-06-30T13-44-00/samples/sample-3200/A\ Detla\ Airlines\ aircraft\ descends\ onto\ the\ runaway\ during\ a\ cloudless\ morning..gif -m data/loveu-tgve/videos/aircraft-landing.mp4
# 32.23, 0.9198

python scripts/recons_score.py -g .log/vae3d/loveu-tgve/base/2024-06-30T13-44-00/samples/sample-3200/Drone\ flyover\ of\ the\ Eiffel\ Tower\ in\ front\ of\ the\ city,\ sunset..gif -m data/loveu-tgve/videos/eiffel-flyover.mp4
# 30.41, 0.8267


python scripts/extract_correspondences.py -d data/few-shot/ski-lift-time-lapse -t 0.018
python scripts/extract_correspondences.py -d data/one-shot/ski-lift-time-lapse -t 0.018



# one-shot
python scripts/check_correspondences.py -p data/one-shot/ski-lift-time-lapse/video_abstract_patches.pkl -a ".log/tp_tive_with_clip/one-shot/ski-lift-time-lapse/base/2024-07-05T15-19-42"

python scripts/fuse_videos.py -s data/one-shot/ski-lift-time-lapse/videos/ski-lift-time-lapse.mp4 -t .output/.log/tp_tive_with_clip/one-shot/ski-lift-time-lapse/base/2024-07-05T15-19-42/ski-lift-time-lapse-CHAIRLIFTS-sim.gif -o .output/.log/tp_tive_with_clip/one-shot/ski-lift-time-lapse/base/2024-07-05T15-19-42/


# few-shot
python scripts/check_correspondences.py -v data/few-shot/ski-lift-time-lapse/videos/ski-lift-time-lapse.mp4 -t "An image of $CHAIRLIFTS." -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"

python scripts/fuse_videos.py -s data/few-shot/ski-lift-time-lapse/videos/ski-lift-time-lapse.mp4 -t .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23/sim.gif -o .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23




python scripts/check_correspondences.py -v data/few-shot/ski-lift-time-lapse/videos/ski-lift-time-lapse.mp4 -t "An image of chairlifts." -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"

python scripts/fuse_videos.py -s data/few-shot/ski-lift-time-lapse/videos/ski-lift-time-lapse.mp4 -t .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23/sim.gif -o .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23



python scripts/check_correspondences.py -v data/few-shot/ski-lift-time-lapse/videos/ski-lift-time-lapse.mp4 -t "A time lapse video of $CHAIRLIFTS moving up and down with a snowy mountain and a blue sky in the background." -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"

python scripts/fuse_videos.py -s data/few-shot/ski-lift-time-lapse/videos/ski-lift-time-lapse.mp4 -t .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23/sim.gif -o .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23



python scripts/check_correspondences.py -v data/few-shot/ski-lift-time-lapse/videos/kettleball-training.mp4 -t "An image of a $WARM sun." -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"

python scripts/fuse_videos.py -s data/few-shot/ski-lift-time-lapse/videos/kettleball-training.mp4 -t .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23/sim.gif -o .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23



python scripts/check_correspondences.py -v data/few-shot/ski-lift-time-lapse/videos/kettleball-training.mp4 -t "An image of a warm sun." -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"

python scripts/fuse_videos.py -s data/few-shot/ski-lift-time-lapse/videos/kettleball-training.mp4 -t .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23/sim.gif -o .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23


# CROSS

python scripts/check_correspondences.py -v data/few-shot/ski-lift-time-lapse/videos/ski-lift-time-lapse.mp4 -t "An image of a $WARM sun." -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"

python scripts/fuse_videos.py -s data/few-shot/ski-lift-time-lapse/videos/ski-lift-time-lapse.mp4 -t .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23/sim.gif -o .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23


python scripts/check_correspondences.py -v data/few-shot/ski-lift-time-lapse/videos/kettleball-training.mp4 -t "An image of chairlifts." -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"

python scripts/fuse_videos.py -s data/few-shot/ski-lift-time-lapse/videos/kettleball-training.mp4 -t .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23/sim.gif -o .output/.log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23



# check abstracts distance
python scripts/check_abstracts_distance.py -s "$CHAIRLIFTS" -t "a $WARM sun" -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"

python scripts/check_abstracts_distance.py -s "chairlifts" -t "a $WARM sun" -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"

python scripts/check_abstracts_distance.py -s "$CHAIRLIFTS" -t "a warm sun" -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"

python scripts/check_abstracts_distance.py -s "chairlifts" -t "a warm sun" -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"

python scripts/check_abstracts_distance.py -s "water" -t "a $WARM sun" -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"

python scripts/check_abstracts_distance.py -s "water" -t "a warm sun" -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-02T14-47-23"


python scripts/check_abstracts_distance.py -s "chairlifts" -t "a $WARM sun" -a ".log/tp_tive_with_clip/few-shot/ski-lift-time-lapse/base/2024-07-04T14-26-40"

# "A man does kettlebell exercises on a beach with a $WARM sun in the background."