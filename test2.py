import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.dataset_utils import TestSpecificDataset
from utils.image_io import save_image_tensor
from net.model2 import PromptIR
from options import options as opt
import lightning.pytorch as pl
import glob
import torch.nn.functional as F
import torch.nn as nn

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.l1_loss = nn.L1Loss()
        self.lambda_l1 = 1.0 
        self.psnr_values = []

    def forward(self, x, de_id=None):
        return self.net(x)

    def calculate_psnr(self, pred, target, data_range=1.0):
        mse = F.mse_loss(pred, target, reduction='mean')
        if mse == 0:
            return float('inf')
        max_value = torch.tensor(data_range, device=pred.device)
        psnr = 20 * torch.log10(max_value) - 10 * torch.log10(mse)
        return psnr.item()

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored, task = self.net(degrad_patch)
        l1_loss = self.l1_loss(restored, clean_patch)
        total_loss = self.lambda_l1 * l1_loss 

        psnr = self.calculate_psnr(restored, clean_patch, data_range=1.0)
        self.psnr_values.append(psnr)

        for b in range(len(task)):
            task_loss = total_loss[b].item() if len(total_loss.shape) > 0 else total_loss.item()
            self.log(f"train_loss_{task[b]}", task_loss)
            self.log(f"train_l1_loss_{task[b]}", l1_loss[b].item() if len(l1_loss.shape) > 0 else l1_loss.item())
            self.log(f"train_psnr_{task[b]}", psnr, on_step=True, on_epoch=False)

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
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=5, max_epochs=opt.epochs, warmup_start_lr=1e-6, eta_min=1e-6)
        return [optimizer], [scheduler]

def test_Derain_Dehaze(net, dataset):
    predictions = {}
    output_path_derain = os.path.join(opt.output_path, "derain")
    output_path_desnow = os.path.join(opt.output_path, "desnow")
    os.makedirs(output_path_derain, exist_ok=True)
    os.makedirs(output_path_desnow, exist_ok=True)

    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=opt.num_workers)

    with torch.no_grad():
        for ([degraded_name], degrad_patch) in tqdm(testloader, desc="Processing images"):
            degrad_patch = degrad_patch.cuda()
            restored, task = net(degrad_patch) 
            print(f"Task detected: {task}")

            restored_np = restored.cpu().numpy().squeeze()
            restored_np = np.clip(restored_np * 255, 0, 255).astype(np.uint8)
            print(f"Restored image shape: {restored_np.shape}")
            predictions[degraded_name[0] + '.png'] = restored_np

            output_path = output_path_derain if task[0] == "derain" else output_path_desnow
            save_image_tensor(restored, os.path.join(output_path, degraded_name[0] + '.png'))
            print(f"Image {degraded_name[0]} detected as {task[0]} noise")

    np.savez(os.path.join(opt.output_path, 'pred.npz'), **predictions)
    print(f"Saved pred.npz to {os.path.join(opt.output_path, 'pred.npz')}")

    return predictions

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(opt.cuda)

    if opt.ckpt_name:
        ckpt_path = os.path.join(opt.ckpt_dir, opt.ckpt_name)
    else:
        ckpt_files = glob.glob(os.path.join(opt.ckpt_dir, "*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in {opt.ckpt_dir}/")
        ckpt_path = max(ckpt_files, key=os.path.getctime)
    print("CKPT name: {}".format(ckpt_path))

    net = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    test_dataset = TestSpecificDataset(opt)
    if opt.mode == 1:
        predictions = test_Derain_Dehaze(net, test_dataset)
    else:
        raise ValueError("Only mode=1 (derain and desnow) is supported for this task.")