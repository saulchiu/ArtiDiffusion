import torch.nn.functional as F
from tools.img import cal_ssim, cal_ppd


def loss_4(p_trigger, trigger, x_p_no_trigger, x_no_trigger):
    loss_p1 = F.mse_loss(x_p_no_trigger, x_no_trigger)
    loss_p2 = 1 - cal_ssim(x_p_no_trigger, x_no_trigger)
    loss_p3 = cal_ppd(trigger, p_trigger)
    return 2 * loss_p1 + 2 * loss_p2 + 4 * loss_p3


loss_dict = {
    4: loss_4
}
