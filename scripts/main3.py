import warnings
warnings.filterwarnings("ignore")

import numpy as np 
import random
import yaml
import torch
from torchvision import transforms
from torchsummary import summary
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

DIR_TRAIN      = config['train_dir']
DIR_TEST       = config['test_dir']
train_split    = config['train_split']
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

# Random seed
SEED = config['SEED']
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
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(10),
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
    train_idx = train_idx[:10000]
print("Train images:", len(train_idx))
print("Validation images:", len(val_idx))

train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
val_sampler   = torch.utils.data.SubsetRandomSampler(val_idx)

# Create Dataloader
cifar_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
cifar_val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)
cifar_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Plot sample image
sample = next(iter(cifar_val_loader))
data_loader.plot_input_sample(batch_data=sample,
                  mean = [0.49139968, 0.48215827 ,0.44653124],
                  std = [0.24703233,0.24348505,0.26158768],
                  to_denormalize = True,
                  figsize = (3,3))


#########################################################
################    train and test     ##################
#########################################################

# Create model
bs = 5
p = 4
n_patches = (32 * 32) // (p * p)
embed_dim = p * p * 3
d_model = 128
n_heads = 8


cnn_model = model.ViT(embed_size=embed_dim, patch_size=4, d_model=128, num_patches=n_patches, 
                        mlp_filters=128, num_heads=8, num_layers=2, 
                        model_dim=128, num_classes=10)

summary(cnn_model, (3, 32, 32))

model_name = "CNNmodel_BlockA"

# # train
# trainer: model, optimizer, report
trainer = train.train(
                    train_loader=cifar_train_loader,
                    val_loader=cifar_val_loader,
                    model=cnn_model,
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
test_acc = evaluate.eval(cifar_test_loader, cnn_model)

print(f"Test accuracy: {test_acc.item() * 100}%")


