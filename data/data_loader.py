
import os
import glob
import yaml
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


def list_image_path(img_dir):
    classes = os.listdir(img_dir)
    print("\nTotal Classes: ", len(classes))

    list_imgs = []

    for _class in classes:
        list_imgs += glob.glob(img_dir + '\\' + _class + '/*.png')
    
    return classes, list_imgs


class CIFAR10Dataset(Dataset):

    def __init__(self, imgs_list, classes, transforms=None):
        super(CIFAR10Dataset, self).__init__()
        self.imgs_list = imgs_list
        self.class_to_int = {classes[i] : i for i in range(len(classes))}
        self.transforms = transforms
        
        
    def __getitem__(self, index):

        image_path = self.imgs_list[index]

        # Reading image
        image = Image.open(image_path)

        # Retriving class label
        label = image_path.split("\\")[-2]
        label = self.class_to_int[label]
        # label = torch.nn.functional.one_hot(label, num_classes=10).float()
        label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=10).float()
        # label = np.eye(10)[label].type(torch.LongTensor)

        # Applying transforms on image
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label


    def __len__(self):
        return len(self.imgs_list)
    
# plot a sample of data
def plot_input_sample(batch_data,
                      mean = [0.49139968, 0.48215827 ,0.44653124],
                      std = [0.24703233,0.24348505,0.26158768],
                      to_denormalize = False,
                      figsize = (3,3)):

    batch_image, lbl = batch_data
    batch_size = batch_image.shape[0]

    random_batch_index = random.randint(0,batch_size-1)
    random_image, random_lbl = batch_image[random_batch_index], lbl[random_batch_index]
    
    image_transposed = random_image.detach().numpy().transpose((1, 2, 0))
    if to_denormalize:
        image_transposed = np.array(std)*image_transposed + np.array(mean)
        image_transposed = image_transposed.clip(0,1)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image_transposed)
    ax.set_title(f"lbl:{random_lbl}")
    ax.set_axis_off()


if __name__ == "__main__":
    print("dataloader")
