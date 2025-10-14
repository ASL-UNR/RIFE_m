import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
from model.IFNet_m import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, local_rank=-1, arbitrary=True):
        if arbitrary == True:
            print("using IFNet_m")
            self.flownet = IFNet_m()
        else:
            self.flownet = IFNet()
            print("using IFNet")
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0, strict=True):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))))
        '''
        """
        Single-GPU friendly loader:
        - Works with both clean keys and old DDP ('module.') keys
        - Restores optimizer and returns (epoch, step) if present
        - Safe on CPU or GPU
        """
        if rank > 0:
            return None, None  # no-op on non-zero ranks (future-proof)

        ckpt_path = os.path.join(path, "flownet.pkl")
        if not os.path.isfile(ckpt_path):
            print(f"[load_model] No checkpoint at {ckpt_path}")
            return None, None

        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Backward-compat: support old files that were weights-only
        state = ckpt.get("model", ckpt)

        # Strip 'module.' if checkpoint came from DDP
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        # Load weights
        self.flownet.load_state_dict(state, strict=strict)

        # Load optimizer if available
        if "optim" in ckpt:
            try:
                self.optimG.load_state_dict(ckpt["optim"])
            except Exception as e:
                print(f"[load_model] Optimizer state not loaded ({e}); continuing with fresh optimizer.")

        epoch = ckpt.get("epoch")
        step = ckpt.get("step")
        print(f"[load_model] Loaded checkpoint from {ckpt_path} (epoch={epoch}, step={step})")
        return epoch, step
        '''

    def save_model(self, path, rank=0, epoch=None, step=None):
        #if rank == 0:
            #torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))
        """
        Single-GPU friendly saver:
        - Ensures folder exists
        - Always saves a clean (no 'module.') state_dict
        - Optionally stores epoch/step to resume later
        """
        # rank guard is harmless on single GPU; keeps compatibility if you later add DDP
        if rank != 0:
            return

        # If you ever wrap with DDP in the future, save underlying module weights
        if isinstance(self.flownet, DDP):
            net = self.flownet.module
            print("DDP")
        else:
            net = self.flownet

        # build a unique filename
        if epoch is not None:
            if step is not None:
                fname = f"flownet_epoch{epoch:04d}_step{step:08d}.pkl"
            else:
                fname = f"flownet_epoch{epoch:04d}.pkl"
        else:
            fname = "flownet.pkl"  # fallback if you don’t pass epoch/step

        ckpt_path = os.path.join(path, fname)
        torch.save(self.flownet.state_dict(), ckpt_path)
        print(f"[save_model] Saved checkpoint to {ckpt_path}")

    def inference(self, img0, img1, timestep, scale=1, scale_list=[4, 2, 1], TTA=False):
        for i in range(3):
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, timestep=timestep, scale=scale_list)
        if TTA == False:
            return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), timestep=timestep, scale=scale_list)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
    
    def forward_and_loss(self, imgs, gt, timestep, training=True):
        """
        Forward pass that computes all losses but does NOT zero_grad or step.
        Returns: (pred, info, loss_G)
        """
        if training:
            self.train()
        else:
            self.eval()

        # Forward
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), timestep, scale=[4, 2, 1])

        # Losses
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        loss_G = loss_l1 + loss_tea + loss_distill * 0.005  # keep same weighting

        # Pack info in same shape you already log
        info = {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            'timestep': timestep
        }
        return merged[2], info, loss_G

    def update(self, imgs, gt, timestep, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        flow, mask, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), timestep, scale=[4, 2, 1])
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + loss_distill * 0.005 # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
        return merged[2], {
            'merged_tea': merged_teacher,
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            'timestep': timestep
            }
