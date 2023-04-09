import warnings
import sys

warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
from tensorfn import load_config as DiffConfig
import numpy as np
from config.diffconfig import DiffusionConfig, get_model_conf
import torch.distributed as dist
import os, glob, cv2, time, shutil
from models.unet_autoenc import BeatGANsAutoencConfig
from diffusion import create_gaussian_diffusion, make_beta_schedule, ddim_steps
import torchvision.transforms as transforms
import torchvision
import pickle


class Predictor():
    def __init__(self):
        """Load the model into memory to make running multiple predictions efficient"""

        conf = DiffConfig(DiffusionConfig, './config/diffusion.conf', show=False)

        self.model = get_model_conf().make_model()
        ckpt = torch.load("checkpoints/last.pt")
        self.model.load_state_dict(ckpt["ema"])
        self.model = self.model.cuda()
        self.model.eval()

        self.betas = conf.diffusion.beta_schedule.make()
        self.diffusion = create_gaussian_diffusion(self.betas, predict_xstart=False)  # .to(device)

        # self.pose_list = glob.glob('data/deepfashion_256x256/target_pose/*.npy')
        self.transforms = transforms.Compose([transforms.Resize((256, 256), interpolation=Image.BICUBIC),
                                              transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5),
                                                                                          (0.5, 0.5, 0.5))])

    def predict_pose(
            self,
            image,
            pose_path,
            sample_algorithm='ddim',
            nsteps=100,

    ):
        """Run a single prediction on the model"""

        src = Image.open(image)
        src = self.transforms(src).unsqueeze(0).cuda()
        # tgt_pose = torch.stack(
        #     [transforms.ToTensor()(np.load(ps)).cuda() for ps in np.random.choice(self.pose_list, num_poses)], 0)
        with open(pose_path, 'rb') as f:
            pose_model = pickle.load(f)
            tgt_pose = pose_model['out_poses']
            tgt_pose = torch.from_numpy(tgt_pose)
            tgt_pose = tgt_pose.unsqueeze(0).cuda()

        # src = src.repeat(num_poses, 1, 1, 1)

        if sample_algorithm == 'ddpm':
            samples = self.diffusion.p_sample_loop(self.model, x_cond=[src, tgt_pose], progress=True, cond_scale=2)
        elif sample_algorithm == 'ddim':
            noise = torch.randn(src.shape).cuda()
            seq = range(0, 1000, 1000 // nsteps)
            xs, x0_preds = ddim_steps(noise, seq, self.model, self.betas.cuda(), [src, tgt_pose])
            samples = xs[-1].cuda()

        samples_grid = torch.cat([src[0], torch.cat([samps for samps in samples], -1)], -1)
        samples_grid = (torch.clamp(samples_grid, -1., 1.) + 1.0) / 2.0
        pose_grid = torch.cat([torch.zeros_like(src[0]), torch.cat([samps[:3] for samps in tgt_pose], -1)], -1)

        output = torch.cat([1 - pose_grid, samples_grid], -2)

        numpy_imgs = output.unsqueeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()
        fake_imgs = (255 * numpy_imgs).astype(np.uint8)
        Image.fromarray(fake_imgs[0]).save('output.png')


if __name__ == "__main__":
    img_path = sys.argv[1]
    pose_path = sys.argv[2]
    obj = Predictor()
    obj.predict_pose(image=img_path, pose_path=pose_path)
