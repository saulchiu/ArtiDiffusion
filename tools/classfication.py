import timm
import detectors
import torch
from tools import dataset
from torch.utils.data.dataloader import DataLoader


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc


if __name__ == '__main__':
    device = "cuda:0"
    net = model = timm.create_model("resnet18_cifar10", pretrained=True).to(device)
    train_data, test_data = dataset.prepare_poisoning_dataset(ratio=0.1,
                                                              mask_path='../resource/badnet/trigger_image.png',
                                                              trigger_path='../resource/badnet/trigger_image_grid.png')
    loader = DataLoader(
        dataset=train_data, batch_size=16, num_workers=4, shuffle=True
    )
    check_accuracy(loader, net, device)
