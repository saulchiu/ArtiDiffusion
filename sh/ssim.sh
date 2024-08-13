#!/bin/bash

cd ../tools
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/gtsrb/20240715133940_sigmoid_700k_min
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/gtsrb/20240629144235_sigmoid_700k_3
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/gtsrb/20240629133605_sigmoid_700k_7
#
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/gtsrb/20240715133940_sigmoid_700k_min
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/gtsrb/20240629144235_sigmoid_700k_3
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/gtsrb/20240629133605_sigmoid_700k_7
#
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/gtsrb/20240715133940_sigmoid_700k_min
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/gtsrb/20240629144235_sigmoid_700k_3
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/gtsrb/20240629133605_sigmoid_700k_7
#
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240716092724_sigmoid_700k_min
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240716092724_sigmoid_700k_min
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240716092724_sigmoid_700k_min
#
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240703064021_sigmoid_700k_3
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240703064021_sigmoid_700k_3
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240703064021_sigmoid_700k_3
#
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240702095648_sigmoid_700k_7
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240702095648_sigmoid_700k_7
#python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 1024 --path ../results/badnet/cifar10/20240702095648_sigmoid_700k_7

python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240715141515_sigmoid_700k_min
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240715141515_sigmoid_700k_min
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240715141515_sigmoid_700k_min

python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240701102049_sigmoid_700k_3
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240701102049_sigmoid_700k_3
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240701102049_sigmoid_700k_3

python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240701102056_sigmoid_700k_7
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240701102056_sigmoid_700k_7
python eval_sandiffusion.py --mode ssim --device cuda:0 --batch 521 --path ../results/badnet/celeba/20240701102056_sigmoid_700k_7


