#!/bin/bash

cd ../tools

python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240716092724_sigmoid_700k_min
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240716092724_sigmoid_700k_min
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240716092724_sigmoid_700k_min

python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240703064021_sigmoid_700k_3
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240703064021_sigmoid_700k_3
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240703064021_sigmoid_700k_3

python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240702095648_sigmoid_700k_7
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240702095648_sigmoid_700k_7
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240702095648_sigmoid_700k_7

