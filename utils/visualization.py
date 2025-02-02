
import matplotlib.pyplot as plt
import pandas as pd


def extract_numerical_value(tensor_str):
    numerical_value = float(tensor_str.split('(')[1].split(')')[0])
    return numerical_value


def plot_history(report_path):
    data = pd.read_csv(report_path)
    data.head()
    data['epoch']
    data.keys()

    tr_report  = data[data["mode"] == "train"]
    val_report = data[data["mode"] == "val"]

    last_tr_batch  = tr_report["batch_index"].max()
    last_val_batch = val_report["batch_index"].max()

    tr_epoch  = tr_report[tr_report.batch_index == last_tr_batch]
    val_epoch = val_report[val_report.batch_index == last_val_batch]

    
    fig, ax = plt.subplots(2, figsize=(10,10))
    plt.suptitle("Train and Validation history")

    ax[0].plot(tr_epoch["avg_train_loss_till_current_batch"].values, label="train")
    ax[0].plot(val_epoch["avg_val_loss_till_current_batch"].values, label="validation")
    ax[0].set_title("Train and Validation loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].grid(axis='y')
    ax[0].legend(loc=0)
    # ax[0].set_ylim((0,1))

    # Apply the function to each element of the Series
    tr_acc = tr_epoch["avg_train_top1_acc_till_current_batch"].apply(extract_numerical_value).values
    val_acc = val_epoch["avg_val_top1_acc_till_current_batch"].apply(extract_numerical_value).values
    
#     ax[1].plot(tr_epoch["avg_train_top1_acc_till_current_batch"].values, label="train")
#     ax[1].plot(val_epoch["avg_val_top1_acc_till_current_batch"].values, label="validation")
    ax[1].plot(tr_acc, label="train")
    ax[1].plot(val_acc, label="validation")
    ax[1].set_title("Train and Validation accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("acc")
#     ax[1].grid(axis='y')
    ax[1].legend(loc=0)
    # ax[1].set_ylim((0,1))

    plt.show()