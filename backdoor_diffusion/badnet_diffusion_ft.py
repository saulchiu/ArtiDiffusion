import ast
import os
import shutil

import hydra
import torch
import torchvision
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from badnet_diffusion_pred_noice import BadDiffusion, BadTrainer
import yaml
from torch.optim.adam import Adam

import sys

sys.path.append('../')
from tools.eval_diffusion import load_result
from tools.time import now
from tools.prepare_data import prepare_bad_data
from tools import tg_bot


@hydra.main(version_base=None, config_path="../config", config_name="default.yaml")
def ft_benign_model(cfg: DictConfig):
    if cfg.poison_pretrain is False:
        return
    device = cfg.device
    ft_cfg = cfg.ft
    ld = torch.load(f'{ft_cfg.path}/result.pth', map_location=device)
    diffusion, _, _, _ = load_result(DictConfig(ld['config']), device)
    cfg.diffusion.image_size = ld['config']['diffusion']['image_size']
    cfg.trainer.train_batch_size = ld['config']['trainer']['train_batch_size']
    diff_cfg = cfg.diffusion
    trainer_cfg = cfg.trainer
    trigger_path = diff_cfg.trigger
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((diff_cfg.image_size, diff_cfg.image_size))
    ])
    trigger = Image.open(trigger_path)
    trigger = transform(trigger)
    trigger = trigger.to(device)

    target_folder = f'../results/ft/{cfg.attack}/{cfg.dataset_name}/{now()}'
    cfg.trainer.results_folder = target_folder
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    trainer = BadTrainer(
        diffusion,
        bad_folder=trainer_cfg.bad_folder,
        good_folder=trainer_cfg.good_folder,
        train_batch_size=trainer_cfg.train_batch_size,
        train_lr=trainer_cfg.train_lr / 1e5,
        train_num_steps=trainer_cfg.train_num_steps,
        gradient_accumulate_every=trainer_cfg.gradient_accumulate_every,
        ema_decay=trainer_cfg.ema_decay,
        amp=trainer_cfg.amp,
        calculate_fid=trainer_cfg.calculate_fid,
        ratio=trainer_cfg.ratio,
        results_folder=target_folder,
        server=trainer_cfg.server,
        save_and_sample_every=trainer_cfg.save_and_sample_every if trainer_cfg.save_and_sample_every > 0 else trainer_cfg.train_num_steps,
    )
    if trainer.accelerator.is_main_process:
        prepare_bad_data(cfg)
    loss_list, fid_list = trainer.ft_train(trigger, 1e-5, "badnet")
    ret = {
        'loss_list': loss_list,
        'fid_list': fid_list,
        'config': OmegaConf.to_object(cfg),
        'diffusion': diffusion.state_dict(),
    }
    torch.save(ret, f'{target_folder}/result.pth')
    script_name = os.path.basename(__file__)
    target_file_path = os.path.join(target_folder, script_name)
    shutil.copy(__file__, target_file_path)
    tg_bot.send2bot(OmegaConf.to_yaml(OmegaConf.to_object(cfg)), trainer_cfg.server)
    print(target_folder)


if __name__ == '__main__':
    ft_benign_model()
