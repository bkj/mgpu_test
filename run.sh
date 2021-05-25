#!/bin/bash

make clean ; make all -j12

INPATH=data/mmio/graph500-scale22-ef16_adj.bin

time CUDA_VISIBLE_DEVICES=0       ./bin/color $INPATH
# time CUDA_VISIBLE_DEVICES=0,1     ./bin/color $INPATH
# time CUDA_VISIBLE_DEVICES=1,2     ./bin/color $INPATH
time CUDA_VISIBLE_DEVICES=0,1,2,3 ./bin/color $INPATH

