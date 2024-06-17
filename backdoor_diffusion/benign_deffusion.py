import argparse
import math

import hydra
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, denoising_diffusion_pytorch
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import divisible_by, num_to_groups, exists
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation
from torchvision import utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append('..')
from tools import tg_bot
from tools.prepare_data import prepare_bad_data
from tools.time import now
from tools.dataset import rm_if_exist


class BenignTrainer(denoising_diffusion_pytorch.Trainer):
    def __init__(self, diffusion, good_folder, train_batch_size, train_lr, train_num_steps,
                 gradient_accumulate_every, results_folder, save_and_sample_every, ema_decay, amp,
                 calculate_fid):
        super().__init__(diffusion_model=diffusion, folder=good_folder, train_batch_size=train_batch_size,
                         train_lr=train_lr, train_num_steps=train_num_steps,
                         gradient_accumulate_every=gradient_accumulate_every, ema_decay=ema_decay, amp=amp,
                         calculate_fid=calculate_fid, results_folder=results_folder,
                         save_and_sample_every=save_and_sample_every)

    def train(self):
        rm_if_exist(f'../runs/{tag}_loss')
        rm_if_exist(f'../runs/{tag}_fid')
        writer1 = SummaryWriter(f'../runs/{tag}_loss')
        writer2 = SummaryWriter(f'../runs/{tag}_fid')
        accelerator = self.accelerator
        device = accelerator.device
        loss_list = []
        fid_list = []
        min_loss = 1e3
        min_fid = 1e3
        fid_evaler = FIDEvaluation(
            batch_size=self.batch_size,
            dl=self.dl,
            sampler=self.ema.ema_model,
            channels=self.channels,
            accelerator=self.accelerator,
            stats_dir=self.results_folder,
            device=self.device,
            num_fid_samples=1000,
            inception_block_idx=2048
        )
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                for mode in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    self.accelerator.backward(loss)
                # use tensorboard
                writer1.add_scalar(tag, float(total_loss), int(self.step))
                pbar.set_description(f'loss: {total_loss:.4f}')
                formatted_loss = format(total_loss, '.4f')
                min_loss = min(min_loss, total_loss)
                loss_list.append(float(formatted_loss))
                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()
                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()
                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()
                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                        fid = fid_evaler.fid_score()
                        writer2.add_scalar(tag, float(fid), int(self.step))
                        min_fid = min(fid, min_fid)
                        fid_list.append(fid)
                        all_images = torch.cat(all_images_list, dim=0)
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'),
                                         nrow=int(math.sqrt(self.num_samples)))
                        if self.calculate_fid and self.step == self.train_num_steps:
                            fid_score = self.fid_scorer.fid_score()
                            fid_list.append(fid_score)
                            min_fid = min(fid_score, min_fid)
                            tg_bot.send2bot(msg=f'min loss: {min_loss};\n min fid: {min_fid}', title=tag)
                            accelerator.print(f'fid_score: {fid_score}')
                writer1.flush()
                writer2.flush()
                pbar.update(1)

        accelerator.print('training complete')
        data = {
            'step': self.step,
            'diffusion': self.accelerator.get_state_dict(self.model),
            'unet': self.model.model.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            "config": task_config,
            "loss_list": loss_list,
            'fid_list': fid_list
        }
        return data


def get_args():
    parser = argparse.ArgumentParser(description='This script does amazing things.')
    parser.add_argument('--batch', type=int, default=128, help='Batch size for processing')
    parser.add_argument('--step', type=int, default=10000, help='Number of steps for the diffusion model')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run the process on (e.g., "cpu" or "cuda:0")')
    parser.add_argument('--results_folder', type=str, default='./results', help='Folder to save results')
    parser.add_argument('--server', type=str, help='which server you use, lab, pc, or lv')
    parser.add_argument('--save_and_sample_every', type=int, default=10000, help='save every step')
    parser.add_help = True
    return parser.parse_args()


from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(config: DictConfig):
    prepare_bad_data(config)
    print(OmegaConf.to_yaml(OmegaConf.to_object(config)))
    unet_config = config.noise_predictor
    diff_config = config.diffusion
    trainer_config = config.trainer
    import os
    import shutil
    script_name = os.path.basename(__file__)
    target_folder = f'../results/{config.attack}/{config.dataset_name}/{now()}'
    dataset_all = f'../dataset/dataset-{config.dataset_name}-all'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    target_file_path = os.path.join(target_folder, script_name)
    shutil.copy(__file__, target_file_path)
    device = diff_config.device
    import os
    os.environ["ACCELERATE_TORCH_DEVICE"] = device
    if config.dataset_name == "cifar10":
        import model.unet
        import model.DDPM
        unet = model.unet.Unet(
            dim=128,
            image_size=32,
            dim_multiply=(1, 2, 2, 2),
            dropout=0.1
        )
        diffusion = model.DDPM.GaussianDiffusion(
            model=unet,
            image_size=32,
            time_step=1000,
            loss_type='l2'
        )
    else:
        unet = Unet(
            dim=unet_config.dim,
            dim_mults=tuple(map(int, unet_config.dim_mults[1:-1].split(', '))),
            flash_attn=unet_config.flash_attn
        )
        diffusion = GaussianDiffusion(
            unet,
            image_size=diff_config.image_size,
            timesteps=diff_config.timesteps,  # number of steps
            sampling_timesteps=diff_config.sampling_timesteps,
            objective=diff_config.objective,
        )

    trainer = BenignTrainer(
        diffusion,
        good_folder=dataset_all,
        train_batch_size=trainer_config.train_batch_size,
        train_lr=trainer_config.train_lr,
        train_num_steps=trainer_config.train_num_steps,
        gradient_accumulate_every=trainer_config.gradient_accumulate_every,
        ema_decay=trainer_config.ema_decay,
        amp=trainer_config.amp,
        calculate_fid=trainer_config.calculate_fid,
        results_folder=target_folder,
        save_and_sample_every=trainer_config.save_and_sample_every if trainer_config.save_and_sample_every > 0 else trainer_config.train_num_steps,
    )
    global task_config
    task_config = OmegaConf.to_object(config)
    global tag
    tag = f'{config.dataset_name}_{config.attack}'
    res = trainer.train()
    if trainer.accelerator.is_main_process:
        torch.save(res, f'{target_folder}/result.pth')
        tg_bot.send2bot(OmegaConf.to_yaml(OmegaConf.to_object(config)), 'over')
        print(target_folder)


if __name__ == '__main__':
    main()
