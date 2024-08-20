
python scripts/extract_masks.py -v data/targets/car-turn/videos/car-turn.mp4 -s jeep
python scripts/extract_masks.py -v data/targets/car-drift/videos/car-drift.mp4 -s car
python scripts/extract_masks.py -v data/targets/capsule-rot/videos/capsule-rot.mp4 -s capsule -t 0.004695
python scripts/extract_masks.py -v data/targets/man-skate/videos/man-skate.mp4 -s man
python scripts/extract_masks.py -v data/targets/usa-flag/videos/usa-flag.mp4 -s flag
python scripts/extract_masks.py -v data/targets/lotus/videos/lotus.mp4 -s lotus

python scripts/extract_masks.py -v data/concepts/lambo/videos/lambo.mp4 -s car
python scripts/extract_masks.py -v data/concepts/cybertrunk/videos/cybertrunk.mp4 -s car
python scripts/extract_masks.py -v data/concepts/ferrari/videos/ferrari.mp4 -s car
python scripts/extract_masks.py -v data/concepts/bmw/videos/bmw.mp4 -s car
python scripts/extract_masks.py -v data/concepts/optimus/videos/optimus.mp4 -s robot -t 0.02
python scripts/extract_masks.py -v data/concepts/neo/videos/neo.mp4 -s man
python scripts/extract_masks.py -v data/concepts/pokeball/videos/pokeball.mp4 -s ball -t 0.03
python scripts/extract_masks.py -v data/concepts/chrysanthemum/videos/chrysanthemum.mp4 -s flower -t 0.03