#!/bin/bash

python3.10 train.py -m model.type=vanilla,resnet model.width_factor=1.0,1.25,1.5 model.depth_factor=1,1.5,2
