import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("utils")
from metrics import AverageMeter, accuracy


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("utils")
# from metrics import AverageMeter, accuracy


def save_model(file_path, file_name, model, optimizer=None):
    """
    In this function, a model is saved. Usually save model after training in each epoch.
    ------------------------------------------------
    Args:
        - model (torch.nn.Module)
        - optimizer (torch.optim)
        - file_path (str): Path(Folder) for saving the model
        - file_name (str): name of the model checkpoint to save
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def train(
    train_loader,
    val_loader,
    model,
    model_name,
    epochs,
    learning_rate,
    gamma,
    step_size,
    device,
    load_saved_model,
    ckpt_save_freq,
    ckpt_save_path,
    ckpt_path,
    report_path,
):

    model = model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimzier
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))

    if load_saved_model:
        model, optimizer = load_model(
            ckpt_path=ckpt_path, model=model, optimizer=optimizer
        )

    lr_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
#     lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
    
    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_train_loss_till_current_batch",
            "avg_train_top1_acc_till_current_batch",
            "avg_val_loss_till_current_batch",
            "avg_val_top1_acc_till_current_batch"])

    for epoch in tqdm(range(1, epochs + 1)):
        top1_acc_train = AverageMeter()
        loss_avg_train = AverageMeter()
        top1_acc_val = AverageMeter()
        loss_avg_val = AverageMeter()

        model.train()
        mode = "train"

        loop_train = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc="train",
            position=0,
            leave=True)
        
        for batch_idx, (images, labels) in loop_train:
            images = images.to(device)
            labels = labels.to(device)
            labels_pred, _ = model(images)
            loss = criterion(labels_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1 = accuracy(labels_pred, labels)
            top1_acc_train.update(acc1, images.size(0))
            loss_avg_train.update(loss.item(), images.size(0))
            
            new_row = pd.DataFrame(
                {"model_name": model_name,
                 "mode": mode,
                 "epoch": epoch,
                 "learning_rate":optimizer.param_groups[0]["lr"],
                 "batch_size": images.size(0),
                 "batch_index": batch_idx,
                 "loss_batch": loss.detach().item(),
                 "avg_train_loss_till_current_batch":loss_avg_train.avg,
                 "avg_train_top1_acc_till_current_batch":top1_acc_train.avg,
                 "avg_val_loss_till_current_batch":None,
                 "avg_val_top1_acc_till_current_batch":None},index=[0])


            report.loc[len(report)] = new_row.values[0]

            loop_train.set_description(f"Train - epoch {epoch}/{epochs}")
#             loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
                top1_accuracy_train="{:.4f}".format(top1_acc_train.avg),
                max_len=2,
                refresh=True,
            )
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=ckpt_save_path,
                file_name=f"ckpt_{model_name}_epoch{epoch}.ckpt",
                model=model,
                optimizer=optimizer,
            )


        model.eval()
        mode = "val"
        with torch.no_grad():
            loop_val = tqdm(
                enumerate(val_loader, 1),
                total=len(val_loader),
                desc="val",
                position=0,
                leave=True,
            )
            for batch_idx, (images, labels) in loop_val:
                optimizer.zero_grad()
                images = images.to(device)
                labels = labels.to(device)
                labels_pred, _ = model(images)
                loss = criterion(labels_pred, labels)
                acc1 = accuracy(labels_pred, labels)
                top1_acc_val.update(acc1, images.size(0))
                loss_avg_val.update(loss.item(), images.size(0))
                new_row = pd.DataFrame(
                    {"model_name": model_name,
                     "mode": mode,
                     "epoch": epoch,
                     "learning_rate":optimizer.param_groups[0]["lr"],
                     "batch_size": images.size(0),
                     "batch_index": batch_idx,
                     "loss_batch": loss.detach().item(),
                     "avg_train_loss_till_current_batch":None,
                     "avg_train_top1_acc_till_current_batch":None,
                     "avg_val_loss_till_current_batch":loss_avg_val.avg,
                     "avg_val_top1_acc_till_current_batch":top1_acc_val.avg},index=[0],)
                    
                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"Val - epoch {epoch}/{epochs}")
#                 loop_val.set_description(f"val - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                    top1_accuracy_val="{:.4f}".format(top1_acc_val.avg),
                    refresh=True,
                )
        lr_scheduler.step()
#     torch.save(model.state_dict(), f"{model_path}/{model_name}.pt")
    report.to_csv(f"{report_path}/{model_name}_report.csv")
    return model, optimizer, report
