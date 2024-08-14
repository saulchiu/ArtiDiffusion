#!/bin/bash

cd ../tools

python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240715141515_sigmoid_700k_min
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240715141515_sigmoid_700k_min
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240715141515_sigmoid_700k_min

python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240701102049_sigmoid_700k_3
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240701102049_sigmoid_700k_3
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240701102049_sigmoid_700k_3

python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240701102056_sigmoid_700k_7
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240701102056_sigmoid_700k_7
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240701102056_sigmoid_700k_7


