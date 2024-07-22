#!/bin/bash

cd ../tools

path_list=(
"../results/badnet/celeba/20240701101746_sigmoid_700k_1"
"../results/badnet/celeba/20240701102049_sigmoid_700k_3"
"../results/badnet/celeba/20240701102056_sigmoid_700k_7"
"../results/badnet/celeba/xxxx_sigmoid_700k_5"
)

# 遍历数组中的每个路径
for path in "${path_list[@]}"; do
  echo "Evaluated $path"
  python eval_sandiffusion.py --mode fid --device cuda:0 --sampler ddim --batch 128 --path "$path"
done

echo "All paths have been processed."
