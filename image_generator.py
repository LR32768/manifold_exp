import pickle
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image
from torchvision import utils
import shutil
from tqdm import tqdm

class ImageGenerator(object):
    def __init__(self, generator_list, man_dim=10, seed=None, device='cuda:0'):
        self.num_classes = len(generator_list)
        self.device = device
        self.G_list = []

        if seed is not None:
            torch.manual_seed(seed)

        print("Loading Generators.")
        for g in generator_list:
            print(f"Loading {g}")
            with open(g, 'rb') as f:
                G = pickle.load(f)['G_ema'].to(device)
                self.G_list.append(G)

        self.man_dim = man_dim
        self.z_dim = G.z_dim

        print(f"Generating intrinsic basis of shape {self.man_dim}x{self.z_dim}")
        self.intr_basis = []
        for i in range(self.num_classes):
            B = torch.randn(self.man_dim, self.z_dim, device=device) / math.sqrt(self.man_dim)
            self.intr_basis.append(B)

    def generate_dataset(self, dir, num_imgs=1000):
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)

        with torch.no_grad():
            for cls in range(self.num_classes):
                G = self.G_list[cls]
                G.eval()
                if not os.path.exists(os.path.join(dir, f'cls{cls}')):
                    os.mkdir(os.path.join(dir, f'cls{cls}'))
                for i in tqdm(range(num_imgs)):
                    z = self.generate_noise(cls)
                    c = None  # class labels (not used in this example)
                    img = G(z, c)  # NCHW, float32, dynamic range [-1, +1]
                    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                    save_path = os.path.join(dir, f"cls{cls}", f"img{i:05d}.png")
                    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(save_path)

    def sample(self, batch=64, beta=0, uncon=False, img_size=224):
        """
        beta means the beta mix parameter, 0 means discrete without mixing, otherwise
        means using softmax of a Gaussian distribution with temperature 1/beta as mixing weights.

        return:
        mixed image of shape (B,C,H,W), pixel
        """
        if beta < 1e-2:
            label = torch.randint(self.num_classes, (batch,)).to(self.device)
            weight = F.one_hot(label) * 1.0
        else:
            weight = torch.randn(batch, self.num_classes).to(self.device)
            weight = (weight/beta).softmax(dim=-1)
            label = weight.argmax(dim=-1)

        #print(weight)
        sample = None
        for cls in range(self.num_classes):
            if uncon: # Unconditional sample
                z = torch.randn(batch, self.z_dim, device=self.device)
            else:
                z = self.generate_noise(cls, num_sample=batch)
            c = None
            G = self.G_list[cls]
            with torch.no_grad():
                imgs = (G(z, c) + 1) * 0.5
                imgs = imgs.unsqueeze(1)
                if sample is None:
                    sample = imgs
                else:
                    sample = torch.cat((sample, imgs), dim=1)

        sample = weight.view(batch, -1, 1, 1, 1) * sample
        sample = sample.sum(dim=1)

        sample = F.interpolate(sample, size=img_size)
        return sample, label

    def generate_noise(self, cls, num_sample=1, perturb=True):
        """ Generate a noise w.r.t. each class's intrinsic basis. """
        sample_z = torch.randn([num_sample, self.man_dim]).to(self.device)
        sample_z = sample_z @ self.intr_basis[cls]
        sample_z = sample_z / sample_z.norm(dim=-1, keepdim=True) * math.sqrt(self.z_dim)
        if perturb:
            sample_z += torch.randn(num_sample, self.z_dim, device=self.device) * 0.02
        return sample_z

    def navigate_manifold(self, out_dir, grid_num=20):
        """
        WARNING: Only support 2 dim intrinsic manifold
        """
        assert self.man_dim == 2

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        with torch.no_grad():
            for cls in range(self.num_classes):
                G = self.G_list[cls]
                G.eval()
                if not os.path.exists(os.path.join(out_dir, f'cls{cls}')):
                    os.mkdir(os.path.join(out_dir, f'cls{cls}'))

                xs = torch.linspace(-3, 3, grid_num)
                ys = torch.linspace(-3, 3, grid_num)
                x, y = torch.meshgrid(xs, ys)
                z = torch.cat((x.reshape(-1,1), y.reshape(-1,1)), dim=-1).to(self.device)

                z = z @ self.intr_basis[cls]
                sample_z = z / z.norm(dim=-1, keepdim=True) * math.sqrt(self.z_dim)
                num_sample = sample_z.size(0)
                for i in tqdm(range(num_sample)):
                    z = sample_z[i].unsqueeze(0)
                    c = None  # class labels (not used in this example)
                    img = G(z, c)  # NCHW, float32, dynamic range [-1, +1]
                    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

                    save_path = os.path.join(out_dir, f"cls{cls}", f"img{i:05d}.png")
                    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(save_path)

if __name__ == "__main__":
    g_list = ['/home/raylu/afhqcat.pkl', '/home/raylu/afhqdog.pkl']
    gen = ImageGenerator(g_list, man_dim=3)

    # gen.generate_dataset(dir='tmp_dataset', num_imgs=100)
    sample, label = gen.sample(batch=8, beta=0.4)
    img = (sample.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)

    for i in range(img.shape[0]):
        PIL.Image.fromarray(img[i].cpu().numpy(), 'RGB').save(f'sample{i}.png')

    print(label)

