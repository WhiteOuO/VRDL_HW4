import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset_utils import PromptTrainDataset
from net.model2 import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset_utils import PromptTrainDataset
from net.model2 import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.nn.functional as F
import os

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.l1_loss = nn.L1Loss()
        self.psnr_values = []

    def forward(self, x, de_id=None):
        return self.net(x)

    def calculate_psnr(self, pred, target, data_range=1.0):
        mse = F.mse_loss(pred, target, reduction='mean')
        if mse == 0:
            return float('inf')
        max_value = torch.tensor(data_range, device=pred.device) 
        psnr = 20 * torch.log10(max_value) - 10 * torch.log10(mse)
        return psnr

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored, task = self.net(degrad_patch)

        l1_loss = self.l1_loss(restored, clean_patch)
        total_loss = l1_loss

        psnr = self.calculate_psnr(restored, clean_patch, data_range=1.0)
        self.psnr_values.append(psnr.item())

        for b in range(len(task)):
            task_loss = total_loss[b].item() if len(total_loss.shape) > 0 else total_loss.item()
            self.log(f"train_loss_{task[b]}", task_loss)
            self.log(f"train_l1_loss_{task[b]}", l1_loss[b].item() if len(l1_loss.shape) > 0 else l1_loss.item())
            self.log(f"train_psnr_{task[b]}", psnr.item(), on_step=True, on_epoch=False)

        return total_loss

    def on_train_epoch_end(self):
        if self.psnr_values:
            avg_psnr = sum(self.psnr_values) / len(self.psnr_values)
            self.log("epoch_avg_psnr", avg_psnr, on_epoch=True, prog_bar=True)
            print(f"Epoch {self.current_epoch} - Average PSNR: {avg_psnr:.2f} dB")
            self.psnr_values = []
        else:
            print(f"Epoch {self.current_epoch} - No PSNR values recorded.")

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=5, max_epochs=opt.epochs, warmup_start_lr=5e-6, eta_min=1e-6)
        return [optimizer], [scheduler]

def main():
    train_transforms = None
    torch.set_float32_matmul_precision('medium')
    print("Options")
    print(opt)
        
    trainset = PromptTrainDataset(opt, transform=train_transforms)
    checkpoint_callback = ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1)
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
        persistent_workers=True
    )
    
    model = PromptIRModel()
    
    ckpt_path = None
    if hasattr(opt, 'ckpt_name') and opt.ckpt_name: 
        ckpt_path = os.path.join(opt.ckpt_dir, opt.ckpt_name)
        print(f"Resuming training from checkpoint: {ckpt_path}")
    
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger(save_dir="logs/"),
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model=model, train_dataloaders=trainloader, ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()