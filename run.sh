#!/bin/bash

make clean ; make all -j12

INPATH=graph500-scale24-ef16_adj.bin

# CUDA_VISIBLE_DEVICES=0       ./bin/test $INPATH
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./bin/test $INPATH

CUDA_VISIBLE_DEVICES=0       ./bin/color_transform $INPATH
CUDA_VISIBLE_DEVICES=0,1,2,3 ./bin/color_transform $INPATH
