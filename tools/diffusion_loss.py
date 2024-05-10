import torch.nn.functional as F
from tools.img import cal_ssim, cal_ppd
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import default, rearrange, random, reduce, extract, cycle, \
    Dataset, divisible_by, num_to_groups


def bad_loss_4(p_trigger, trigger, x_p_no_trigger, x_no_trigger, factor_list=None):
    if factor_list is None:
        factor_list = [2, 2, 5]
    dev = trigger.device
    p_trigger = p_trigger.to(dev)
    x_p_no_trigger = x_p_no_trigger.to(dev)
    x_no_trigger = x_no_trigger.to(dev)
    loss_p1 = F.mse_loss(x_p_no_trigger, x_no_trigger, reduction='none')
    loss_p2 = 1 - cal_ssim(x_p_no_trigger, x_no_trigger)
    loss_p3 = cal_ppd(trigger, p_trigger)
    return factor_list[0] * loss_p1 + factor_list[1] * loss_p2 + factor_list[2] * loss_p3


loss_dict = {
    4: bad_loss_4
}
