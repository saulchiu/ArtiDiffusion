#!/bin/bash

cd ../tools

path_list=(
"../results/badnet/gtsrb/20240723190050_linear_700k_min"
"../results/badnet/gtsrb/20240724133451_linear_700k_3"
"../results/badnet/gtsrb/20240724191846_linear_700k_7"
)

for path in "${path_list[@]}"; do
  echo "Evaluated $path"
  python eval_sandiffusion.py --mode fid --device cuda:0 --sampler ddim --batch 128 --path "$path"
done

echo "All paths have been processed."
