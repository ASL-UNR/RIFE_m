
import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from model.RIFE import Model
from dataset import *
from dataset import CustomDataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

#device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_learning_rate(step):
    if step < 1400:
        mul = step / 1400.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 1400) / (args.epoch * args.step_per_epoch - 1400.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model, local_rank, args):
    log_path = args.logpath
    print(log_path)
    run_name = time.strftime("rife_%Y%m%d-%H%M%S")
    tb_root = os.path.join(log_path, "tb", run_name)
    os.makedirs(tb_root, exist_ok=True)

    if args.no_ddp or (not args.no_ddp and local_rank == 0):
        writer = SummaryWriter(log_dir=os.path.join(tb_root, "train"))
        writer_val = SummaryWriter(log_dir=os.path.join(tb_root, "validate"))
    else:
        writer = None
        writer_val = None

    if writer is not None:
        layout = {
            "Training/Losses": {
                "losses": ["Multiline", ["loss/l1", "loss/tea", "loss/distill"]]
            },
            "Training/Timestep": {
                "timestep": ["Multiline", ["timestep/mean", "timestep/std"]]
            },
            "Validation/PSNR": {
                "psnr_lines": ["Multiline", ["psnr", "psnr_teacher"]]
            },
        }
        writer.add_custom_scalars(layout)

    step = 0
    nr_eval = 0
    #dataset = VimeoDataset('train')
    #sampler = DistributedSampler(dataset)
    #train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    dataset = CustomDataset('train') # Added
    #sampler = DistributedSampler(dataset) # Added
    sampler = DistributedSampler(dataset) if not args.no_ddp else None
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True, drop_last=True, sampler=sampler) # Added, num workers was 8
    args.step_per_epoch = train_data.__len__()
    #dataset_val = VimeoDataset('validation')
    #val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8)
    dataset_val = CustomDataset('validation') # Added
    val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8) # Added
    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for i, dat in enumerate(train_data):
            learning_rate = get_learning_rate(step)

            for param_group in model.optimG.param_groups:
                param_group['lr'] = learning_rate

            model.optimG.zero_grad()
            running_loss_l1 = 0.0
            running_loss_tea = 0.0
            running_loss_distill = 0.0
            pred_to_log = None
            info_to_log = None

            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            for num_sample in range(4):
                data = dat[num_sample]
                time_stamp = time.time()
                data_gpu, timestep = data
                data_gpu = data_gpu.to(device, non_blocking=True).float() / 255.
                timestep = timestep.to(device=device, dtype=torch.float32, non_blocking=True)
                imgs = data_gpu[:, :6]
                gt = data_gpu[:, 6:9]
        
                #pred, info = model.update(imgs, gt, timestep, learning_rate, training=True) # pass timestep if you are training RIFEm
                #train_time_interval = time.time() - time_stamp
                #time_stamp = time.time()

                pred, info, loss_G = model.forward_and_loss(imgs, gt, timestep, training=True)
                loss_G = loss_G / 4.0
                loss_G.backward()

                running_loss_l1 += (info['loss_l1'].item() / 4.0)
                running_loss_tea += (info['loss_tea'].item() / 4.0)
                running_loss_distill += (info['loss_distill'].item() / 4.0)
                pred_to_log = pred
                info_to_log = info

            #torch.nn.utils.clip_grad_norm_(model.flownet.parameters(), 1.0)
            model.optimG.step()
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()

            if step % 200 == 1 and (args.no_ddp or (not args.no_ddp and local_rank == 0)):
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', running_loss_l1, step)
                writer.add_scalar('loss/tea', running_loss_tea, step)
                writer.add_scalar('loss/distill', running_loss_distill, step)
                writer.add_scalar('timestep/mean', timestep.mean().item(), step)
                writer.add_scalar('timestep/std', timestep.std().item(), step)
                writer.add_histogram('timestep/batch', timestep.detach().cpu(), step)

            if step % 1000 == 1 and (args.no_ddp or (not args.no_ddp and local_rank == 0)):
                gt_np = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask_np = (torch.cat((info_to_log['mask'], info_to_log['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred_np = (pred_to_log.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_np = (info_to_log['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = info_to_log['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info_to_log['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()

                for ind in range(5):
                    imgs_vis = np.concatenate((merged_np[ind], pred_np[ind], gt_np[ind]), 1)[:, :, ::-1]
                    writer.add_image(str(ind) + '/img', imgs_vis, step, dataformats='HWC')
                    writer.add_image(str(ind) + '/flow', np.concatenate((flow2rgb(flow0[ind]), flow2rgb(flow1[ind])), 1), step, dataformats='HWC')
                    writer.add_image(str(ind) + '/mask', mask_np[ind], step, dataformats='HWC')

                writer.flush()
            if args.no_ddp or local_rank == 0:
                    print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, running_loss_l1))
            step += 1

        nr_eval += 1
        if nr_eval % 5 == 0:
            evaluate(model, val_data, step, local_rank, writer_val, args)
        model.save_model(log_path, local_rank, epoch, step)   
        if not args.no_ddp: 
            dist.barrier()

def evaluate(model, val_data, nr_eval, local_rank, writer_val, args):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    for i, dat in enumerate(val_data):
        num_sample = random.randint(0, 3)
        data = dat[num_sample]
        data_gpu, timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.
        timestep = timestep.to(device=device, dtype=torch.float32, non_blocking=True)        
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, info = model.update(imgs, gt, timestep, training=False)
            merged_img = info['merged_tea']
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_distill_list.append(info['loss_distill'].cpu().numpy())
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        if i == 0 and (args.no_ddp or (not args.no_ddp and local_rank == 0)):
            for j in range(10):
                imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j]), nr_eval, dataformats='HWC') #flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')
                writer_val.add_histogram('timestep/batch', timestep.detach().cpu(), nr_eval)
    
    eval_time_interval = time.time() - time_stamp

    if not args.no_ddp and local_rank != 0:
        return
    
    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_ddp', action='store_true', help='Disable DDP for single-GPU mode (e.g., Colab)')
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--logpath', dest='logpath', type=str, required=True)
    args = parser.parse_args()
    logp = args.logpath
    if not os.path.exists(logp):
        os.makedirs(logp)
        print(f"Created directory: {logp}")
    else:
        print(f"Directory already exists: {logp}")

    if not torch.distributed.is_available() or torch.cuda.device_count() <= 1:
        args.no_ddp = True
    if not args.no_ddp:
        torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    #model = Model(args.local_rank)
    model = Model(-1) if args.no_ddp else Model(args.local_rank)
    train(model, args.local_rank, args)
