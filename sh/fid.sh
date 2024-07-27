#!/bin/bash

cd ../tools

path_list=(
"../results/blended/gtsrb/20240701222918_sigmoid_700k_5"
"../results/blended/gtsrb/20240630234646_sigmoid_700k_7"
"../results/blended/cifar10/20240719110926_sigmoid_700k_min"
"../results/blended/cifar10/20240704193814_sigmoid_700k_1"
"../results/blended/cifar10/20240706193250_sigmoid_700k_3"
"../results/blended/cifar10/20240706213026_sigmoid_700k_5"
"../results/blended/cifar10/20240707102600_sigmoid_700k_7"
)

# 遍历数组中的每个路径
for path in "${path_list[@]}"; do
  echo "Evaluated $path"
  python eval_sandiffusion.py --mode fid --device cuda:0 --sampler ddim --batch 128 --path "$path"
done

echo "All paths have been processed."
