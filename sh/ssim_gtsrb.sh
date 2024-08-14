#!/bin/bash

cd ../tools
python eval_sandiffusion.py --mode ssim --device cuda:1 --batch 1024 --path ../results/badnet/gtsrb/20240715133940_sigmoid_700k_min
python eval_sandiffusion.py --mode ssim --device cuda:1 --batch 1024 --path ../results/badnet/gtsrb/20240629144235_sigmoid_700k_3
python eval_sandiffusion.py --mode ssim --device cuda:1 --batch 1024 --path ../results/badnet/gtsrb/20240629133605_sigmoid_700k_7

python eval_sandiffusion.py --mode ssim --device cuda:1 --batch 1024 --path ../results/badnet/gtsrb/20240715133940_sigmoid_700k_min
python eval_sandiffusion.py --mode ssim --device cuda:1 --batch 1024 --path ../results/badnet/gtsrb/20240629144235_sigmoid_700k_3
python eval_sandiffusion.py --mode ssim --device cuda:1 --batch 1024 --path ../results/badnet/gtsrb/20240629133605_sigmoid_700k_7

python eval_sandiffusion.py --mode ssim --device cuda:1 --batch 1024 --path ../results/badnet/gtsrb/20240715133940_sigmoid_700k_min
python eval_sandiffusion.py --mode ssim --device cuda:1 --batch 1024 --path ../results/badnet/gtsrb/20240629144235_sigmoid_700k_3
python eval_sandiffusion.py --mode ssim --device cuda:1 --batch 1024 --path ../results/badnet/gtsrb/20240629133605_sigmoid_700k_7


