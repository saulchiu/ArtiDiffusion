import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import lightning as L

import sys

sys.path.append("../")
from tools.dataset import prepare_poisoning_dataset
from models.resnet import ResNet18


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


class MyLightningModule(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_p = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_p, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        return optimizer

if __name__ == '__main__':
    device = "cuda:0"
    net = ResNet18(num_classes=10).to(device)
    batch, nw = 32, 2
    mask_path = '../resource/badnet/trigger_image.png'
    trigger_path = '../resource/badnet/trigger_image_grid.png'
    train_dataset, test_dataset = prepare_poisoning_dataset(ratio=1e-1, mask_path=mask_path, trigger_path=trigger_path)
    train_loader = DataLoader(
        dataset=train_dataset, shuffle=True, batch_size=batch, num_workers=nw
    )
    test_loader = DataLoader(
        dataset=test_dataset, shuffle=True, batch_size=batch, num_workers=nw
    )
    model = MyLightningModule(model=net)
    trainer = L.Trainer(max_epochs=100, devices=[0])
    # trainer.fit(model=model, train_dataloaders=train_loader)
    ld = torch.load('../models/checkpoint/epoch=99-step=156300.ckpt', map_location=device)
    model.load_state_dict(ld['state_dict'])
    check_accuracy(test_loader, model.model, device)

