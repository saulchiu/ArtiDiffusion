import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import lightning as L

import sys

sys.path.append("../")
from tools.dataset import prepare_poisoning_dataset
from models.resnet import ResNet18


class MyLightningModule(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_p = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_p, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_p = self.forward(x)
        test_loss = torch.nn.functional.cross_entropy(y_p, y)
        self.log('test_loss', test_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_p = self.forward(x)
        val_loss = torch.nn.functional.cross_entropy(y_p, y)
        self.log('val_loss', val_loss)


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
    train_set_size = int(len(train_dataset) * 0.8)
    valid_set_size = len(train_dataset) - train_set_size
    seed = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, valid_set_size],
                                                                 generator=seed)
    train_loader = DataLoader(
        dataset=train_dataset, shuffle=True, batch_size=batch, num_workers=nw
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, shuffle=True, batch_size=batch, num_workers=nw
    )
    test_loader = DataLoader(
        dataset=test_dataset, shuffle=True, batch_size=batch, num_workers=nw
    )
    model = MyLightningModule(model=net)
    trainer = L.Trainer(max_epochs=100, devices=[0])
    trainer.fit(model, train_loader, valid_loader,)

