#!/bin/bash

python3.10 train.py -m model.type=vanilla,resnet width_factor=0.5,1.0,1.25,1.5 depth_factor=0.5,1,1.5,2
