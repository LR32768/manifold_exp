import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from image_generator import ImageGenerator

transform_aug = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

def CatDogDataset(man_dim=5, num_images=100, out_dir='./tmp', use_aug=False, seed=None, device='cuda:0'):
    g_list = ['/home/raylu/afhqcat.pkl', '/home/raylu/afhqdog.pkl']
    gen = ImageGenerator(g_list, man_dim=man_dim, seed=seed, device=device)

    gen.generate_dataset(dir=out_dir, num_imgs=num_images)
    tsfm = transform_aug if use_aug else transform
    dataset = ImageFolder(root=out_dir, transform=tsfm)

    return gen, dataset

if __name__ == "__main__":
    dataset = CatDogDataset(seed=2333)
    print(dataset[0])