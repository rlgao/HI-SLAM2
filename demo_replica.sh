#!/bin/bash

scene="room1"

echo "Running demo for Replica scene: ${scene}"

python demo.py \
    --imagedir /home/grl/datasets/replica/${scene}/colors \
    --calib calib/replica.txt \
    --config config/replica_config.yaml \
    --output outputs/replica/${scene} #\
    # --gsvis \
    # --droidvis 
