#!/bin/bash

# Script to pre-train diffusion behavior policy

# Multi-GPU training available
GPU_LIST=(0)

env_list=(
  "hopper-medium-expert-v2"
  "hopper-medium-v2"
  "hopper-medium-replay-v2"
  "walker2d-medium-expert-v2"
  "walker2d-medium-v2"
  "walker2d-medium-replay-v2"
  "halfcheetah-medium-expert-v2"
  "halfcheetah-medium-v2"
  "halfcheetah-medium-replay-v2"
  "antmaze-umaze-v2"
  "antmaze-umaze-diverse-v2"
  "antmaze-medium-play-v2"
  "antmaze-medium-diverse-v2"
  "antmaze-large-play-v2"
  "antmaze-large-diverse-v2"
)

seed_list=(0)

task=0

for seed in ${seed_list[*]}; do
  for env in ${env_list[*]}; do
    GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
    CUDA_VISIBLE_DEVICES=$GPU_DEVICE python pretrain_behavior.py \
      --env $env \
      --seed $seed
    let "task=$task+1"
  done
done