import warnings
warnings.filterwarnings("ignore")

import numpy as np 
import random
import yaml
import torch
from torch import nan_to_num
from torchvision import transforms
from sklearn.model_selection import train_test_split

import sys
sys.path.append("data")
sys.path.append("models")
sys.path.append("utils")
import data_loader
import model
import train
import evaluate
import visualization


#########################################################
################  Read and set configs  #################
#########################################################

with open(r'config\config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
# partition=True If you want to work on a small part of the data (Including 10000 samples for train and 500 for validation)

DIR_TRAIN      = config['DIR_TRAIN']
DIR_TEST       = config['DIR_TEST']
partition      = config['partition']
batch_size     = config['batch_size']
epochs         = config['epochs']
learning_rate  = config['learning_rate']
gamma          = config['gamma']
step_size      = config['step_size']
ckpt_save_freq = config['ckpt_save_freq']
ckpt_save_path = config['ckpt_save_path']
ckpt_path      = config['ckpt_path']
report_path    = config['report_path']

image_size     = config['image_size']
patch_size     = config['patch_size']
d_model        = config['d_model']
d_hidden       = config['d_hidden']
num_heads      = config['num_heads']
num_layers     = config['num_layers']
num_classes    = config['num_classes']
mlp_filters    = config['mlp_filters']
n_patches      = (image_size ** 2) // (patch_size ** 2)
embed_dim      = 3 * patch_size ** 2

# Random seed
SEED = config['SEED']
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


print("cuda available?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################################################
################  Read and set configs  #################
#########################################################
SEED = 123
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

print("cuda available?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################################################
################  Data preprocessing   ##################
#########################################################

# Load data
classes, train_imgs = data_loader.list_image_path(DIR_TRAIN)
print("Total train images: ", len(train_imgs))

classes, test_imgs = data_loader.list_image_path(DIR_TEST)
print("Total test images: ", len(test_imgs))

# Define transforms
cifar_transforms_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,1.5),hue=(-0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])

cifar_transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])

# Create Datasets
train_dataset = data_loader.CIFAR10Dataset(imgs_list = train_imgs, classes = classes, transforms = cifar_transforms_train)
test_dataset = data_loader.CIFAR10Dataset(imgs_list = test_imgs, classes = classes, transforms = cifar_transforms_test)

# Train and Validation split
cls = {classes[i] : i for i in range(len(classes))}
trainlbl = []
for img in train_imgs:
    # trainlbl.append(cls[img.split('/')[-2]])
    trainlbl.append(cls[img.split('\\')[-2]]) 

train_idx, val_idx= train_test_split(
                            np.arange(len(trainlbl)),
                            test_size=0.2,
                            shuffle=True,
                            stratify=trainlbl)
# partition=True If you want to work on a small part of the data (Including 10000 samples for train and 500 for validation)
if partition:
    val_idx   = val_idx[:50]
    train_idx = train_idx[:1000]
print("Train images:", len(train_idx))
print("Validation images:", len(val_idx))

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_sampler   = torch.utils.data.SubsetRandomSampler(val_idx)

# Create Dataloader
cifar_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
cifar_val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)
cifar_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


#########################################################
################    train and test     ##################
#########################################################

# Create model

ViTmodel = model.ViT(embed_size=embed_dim, patch_size=patch_size, d_model=d_model, d_hidden=d_hidden, num_patches=n_patches,
               mlp_filters=mlp_filters, num_heads=num_heads, num_layers=num_layers, num_classes=num_classes)

# summary(ViTmodel.to("cuda"), (3, 32, 32))

model_name = "ViT"

# # train
# trainer: model, optimizer, report

trainer = train.train(
                    train_loader=cifar_train_loader,
                    val_loader=cifar_val_loader,
                    model=ViTmodel,
                    model_name=model_name,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    step_size=step_size,
                    device=device,
                    load_saved_model=False,
                    ckpt_save_freq=ckpt_save_freq,
                    ckpt_save_path=ckpt_save_path,
                    ckpt_path=ckpt_path,
                    report_path=report_path,
                )


# # Visualize result on training and validation

report_path = f"{report_path}/{model_name}_report.csv"
visualization.plot_history(report_path)

# # Test
test_loss, test_accuracy, attention_maps = evaluate.eval(ViTmodel, cifar_test_loader,  nn.CrossEntropyLoss(), device)

## attention maps
layer_indices = [0, 1, 3]  # Layers 1, 2, and 4
head_indices = [0]  # One head
images = [test_dataset[0][0], test_dataset[1][0]]  # Two images from the test dataset
# images = [test_dataset[0][0]]  # Two images from the test dataset

ViTmodel.eval()
with torch.no_grad():
    test_images = torch.stack(images).to(device)
    outputs, attention_maps = ViTmodel(test_images)

# Plot the attention maps
np = 8
visualization.plot_attention_maps(attention_maps, layer_indices, head_indices, images, np)
