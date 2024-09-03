# Preprocess Data

## Extracting targets mask 
python scripts/extract_masks.py -v data/targets/car-turn/videos/car-turn.mp4 -s jeep
python scripts/extract_masks.py -v data/targets/car-drift/videos/car-drift.mp4 -s car
python scripts/extract_masks.py -v data/targets/forest-drift/videos/forest-drift.mp4 -s car
python scripts/extract_masks.py -v data/targets/capsule-rot/videos/capsule-rot.mp4 -s capsule -t 0.004695
python scripts/extract_masks.py -v data/targets/man-skate/videos/man-skate.mp4 -s man
python scripts/extract_masks.py -v data/targets/usa-flag/videos/usa-flag.mp4 -s flag
python scripts/extract_masks.py -v data/targets/lotus/videos/lotus.mp4 -s lotus
python scripts/extract_masks.py -v data/targets/rain-lotus/videos/rain-lotus.mp4 -s lotus -t 0.015
python scripts/extract_masks.py -v data/targets/gray-dog/videos/gray-dog.mp4 -s dog
python scripts/extract_masks.py -v data/targets/swan/videos/swan.mp4 -s swan -t 0.08
python scripts/extract_masks.py -v data/targets/kitten/videos/kitten.mp4 -s cat
python scripts/extract_masks.py -v data/targets/steel/videos/steel.mp4 -s ball -t 0.015
python scripts/extract_masks.py -v data/targets/kun/videos/kun.mp4 -s basketball -t 0.015
python scripts/extract_masks.py -v data/targets/rain-lotus/videos/rain-lotus.mp4 -s lotus
python scripts/extract_masks.py -v data/targets/tesla/videos/tesla.mp4 -s car

python scripts/extract_masks.py -v data/targets/mallard/videos/mallard.mp4 -s mallard -t 0.03
python scripts/extract_masks.py -v data/targets/bear/videos/bear.mp4 -s bear -t 0.025
python scripts/extract_masks.py -v data/targets/race/videos/race.mp4 -s car -t 0.05
python scripts/extract_masks.py -v data/targets/soccerball/videos/soccerball.mp4 -s soccerball -t 0.03
python scripts/extract_masks.py -v data/targets/tennis/videos/tennis.mp4 -s man -t 0.03


## Extract concepts mask
python scripts/extract_masks.py -v data/concepts/lambo/videos/lambo.mp4 -s car
python scripts/extract_masks.py -v data/concepts/cybertrunk/videos/cybertrunk.mp4 -s car
python scripts/extract_masks.py -v data/concepts/ferrari/videos/ferrari.mp4 -s car
python scripts/extract_masks.py -v data/concepts/bmw/videos/bmw.mp4 -s car
python scripts/extract_masks.py -v data/concepts/optimus/videos/optimus.mp4 -s robot -t 0.02
python scripts/extract_masks.py -v data/concepts/neo/videos/neo.mp4 -s man
python scripts/extract_masks.py -v data/concepts/pokeball/videos/pokeball.mp4 -s ball -t 0.015
python scripts/extract_masks.py -v data/concepts/chrysanthemum/videos/chrysanthemum.mp4 -s flower -t 0.03
python scripts/extract_masks.py -v data/concepts/star/videos/star.mp4 -s star -t 0.015
python scripts/extract_masks.py -v data/concepts/pokeflag/videos/pokeflag.mp4 -s flag -t 0.015
python scripts/extract_masks.py -v data/concepts/savior/videos/savior.mp4 -s man
python scripts/extract_masks.py -v data/concepts/fox/videos/fox.mp4 -s fox -t 0.015
python scripts/extract_masks.py -v data/concepts/mallard/videos/mallard.mp4 -s mallard -t 0.015
python scripts/extract_masks.py -v data/concepts/cat/videos/cat.mp4 -s cat -t 0.015

python scripts/extract_masks.py -v data/concepts/audi/videos/audi.mp4 -s car
python scripts/extract_masks.py -v data/concepts/mercedes/videos/mercedes.mp4 -s car
python scripts/extract_masks.py -v data/concepts/porsche/videos/porsche.mp4 -s car
python scripts/extract_masks.py -v data/concepts/goldentiger/videos/goldentiger.mp4 -s tiger
python scripts/extract_masks.py -v data/concepts/pelican/videos/pelican.mp4 -s pelican
python scripts/extract_masks.py -v data/concepts/rhino/videos/rhino.mp4 -s rhino
python scripts/extract_masks.py -v data/concepts/ultron/videos/ultron.mp4 -s robot
python scripts/extract_masks.py -v data/concepts/ogsoccerball/videos/ogsoccerball.mp4 -s soccerball
python scripts/extract_masks.py -v data/concepts/cow/videos/cow.mp4 -s cow -t 0.03



## Extract GIFs
python scripts/extract_gif.py data/targets/car-turn/videos/car-turn.mp4 6 8 assets/jeep-unet-full-supvis/car-turn.gif

python scripts/extract_gif.py data/targets/man-skate/videos/man-skate.mp4 6 8 assets/man-skate-unet-full-supvis/man-skate.gif

python scripts/extract_gif.py data/targets/tesla/videos/tesla.mp4 6 8 assets/tesla-unet-full-supvis/tesla.gif


## Check clip-score, fc-score, mpsnr
python scripts/check_fc_score.py -v "assets/jeep-unet-full-supvis/to-\$LAMBO.gif"

python scripts/check_mpsnr.py -v1 assets/jeep-unet-full-supvis/car-turn.gif -v2 assets/jeep-unet-full-supvis/to-\$CYBERTRUCK.gif -m data/targets/car-turn/masks/car-turn_jeep_frame_masks.mp4

## Check word in tokenizer
python scripts/check_word_in_tokenizer.py -w Shiba-Inu

## To H264 MP4
python scripts/to_h264_mp4.py -i assets/jeep-unet-full-supvis/concat.mp4 -o assets/jeep-unet-full-supvis/concat.mp4