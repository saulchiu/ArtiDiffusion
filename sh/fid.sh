#!/bin/bash

cd ../tools

path_list=(
"../results/blended/celeba/20240731003421_sigmoid_700k_3"
)

for path in "${path_list[@]}"; do
  echo "Evaluated $path"
  python eval_sandiffusion.py --mode fid --device cuda:0 --sampler ddim --batch 128 --path "$path"
done

echo "All paths have been processed."
