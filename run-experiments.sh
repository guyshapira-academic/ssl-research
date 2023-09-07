#!/bin/bash

python3.10 train.py -m model.type=resnet model.width_factor=1.5 model.depth_factor=2.0
python3.10 train.py -m model.type=vanilla model.width_factor=1.5 model.depth_factor=1.0,2.0
python3.10 train.py -m model.type=vanilla model.width_factor=1.0 model.depth_factor=2.0
