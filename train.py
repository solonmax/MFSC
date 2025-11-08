import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import sys
import argparse
from data import create_dataset
from data.universal_dataset import AlignedDataset_all
from src.model import (ResidualDiffusion, Trainer, Unet, UnetRes, set_seed)
import torch
import torch.nn as nn
from datetime import datetime

def print_network(arg, net):
    """Print the str and parameter number of a network.

    Args:
        net (nn.Module)
    """
    net_cls_str = f'{net.__class__.__name__}'

    net_str = str(net)
    # net_params = sum(map(lambda x: x.numel(), net.parameters()))
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(net_str)
    # print(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f} M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f} M)")
    save_model_log(net, save_dir=arg.model_log_path)


def save_model_log(model: nn.Module, save_dir: str = "model_log.txt"):
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(save_dir, current_time)
    with open(save_path, 'w') as f:
        # 保存模型结构
        f.write("Model Architecture:\n")
        f.write(str(model) + "\n\n")

        # 保存总参数量（可训练）
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total Trainable Parameters: {total_params}\n\n")

        # 保存每个参数的详细信息
        f.write("Parameter Details:\n")
        for name, param in model.named_parameters():
            f.write(f"{name}: shape={tuple(param.shape)}, requires_grad={param.requires_grad}\n")

    print(f"Model log saved to {save_path}")



def parsr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default='/root/private_data/Datasets/Restoration')
    parser.add_argument("--phase", type=str, default='train')
    parser.add_argument("--max_dataset_size", type=int, default=float("inf"))
    parser.add_argument('--load_size', type=int, default=268, help='scale images to this size') #572,268
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
    parser.add_argument('--preprocess', type=str, default='crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument("--bsize", type=int, default=2)
    parser.add_argument("--model_log_path", type=str, default='./logs')
    opt = parser.parse_args()
    return opt

sys.stdout.flush()
set_seed(10)

save_and_sample_every = 1000
if len(sys.argv) > 1:
    sampling_timesteps = int(sys.argv[1])
else:
    sampling_timesteps = 10

train_batch_size = 10
num_samples = 1
sum_scale = 0.01
image_size = 256
condition = True
opt = parsr_args()

results_folder = "./ckpt_universal/diffuir_msfsca10e-4"

if 'universal' in results_folder:
    dataset_fog = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='fog')
    dataset_light = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='light_only')
    dataset_rain = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='rain')
    dataset_snow = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='snow')
    dataset_blur = AlignedDataset_all(opt, image_size, augment_flip=True, equalizeHist=True, crop_patch=True, generation=False, task='blur')
    dataset = [dataset_fog, dataset_light, dataset_rain, dataset_snow, dataset_blur]
    
    num_unet = 1
    objective = 'pred_res'
    test_res_or_noise = "res"
    train_num_steps = 300000
    train_batch_size = 10
    sum_scale = 0.01
    delta_end = 1.8e-3


model = UnetRes(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    num_unet=num_unet,
    condition=condition,
    objective=objective,
    test_res_or_noise = test_res_or_noise,
    uncertainty=True,
)
# model = UnetRes(
#     dim=32,
#     dim_mults=(1, 1, 1, 1),
#     num_unet=num_unet,
#     condition=condition,
#     objective=objective,
#     test_res_or_noise = test_res_or_noise
# )
diffusion = ResidualDiffusion(
    model,
    image_size=image_size,
    timesteps=1000,           # number of steps
    delta_end = delta_end,
    sampling_timesteps=sampling_timesteps,
    objective=objective,
    loss_type='l1',            # L1 or L2
    condition=condition,
    sum_scale=sum_scale,
    test_res_or_noise = test_res_or_noise,
)

print_network(opt, diffusion)



trainer = Trainer(
    diffusion,
    dataset,
    opt,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=8e-5,
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    results_folder = results_folder,
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    num_unet=num_unet,
)

# train
# trainer.load(30)
trainer.train()
