import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def read_test_acc():
    path_output = "../output/"
    dir_output = os.listdir(path_output)
    for dir in dir_output:
        if os.path.isdir(path_output+dir):
            for file in os.listdir(path_output+dir):
                if file.split(".")[-1] == "txt":
                    with open(path_output+dir+'/'+file, "r") as f:
                        acc = round(float(f.readline()) * 100, 2)
                    print(str(dir)+": ", str(acc))


def transition_matrix_visual(path_save="../output/T.png"):
    dist_cols = 2
    dist_rows = 2
    plt.figure(figsize=(5 * dist_cols, 5 * dist_rows))
    plot_idx = 1
    for flip_type in ["symmetric", "pair"]:
        if flip_type == "symmetric":
            noise_rate = "0.2"
        else:
            noise_rate = "0.45"
        path = '../output/mnist_' + flip_type + '_' + noise_rate
        path_T = path + '/T.npy'
        path_T_hat = path + '/T_hat.npy'
        T = np.load(path_T)
        T_hat = np.load(path_T_hat)
        T = np.round(T, decimals=2)
        T_hat = np.round(T_hat, decimals=2)

        ax = plt.subplot(dist_rows, dist_cols, plot_idx)
        ax = sns.heatmap(data=T, cmap='gray', annot=True, annot_kws={'size': 8})
        ax.set_title("Ture noise transition matri\n"+flip_type + '-' + noise_rate)

        ax = plt.subplot(dist_rows, dist_cols, plot_idx+1)
        ax = sns.heatmap(data=T_hat, cmap='gray', annot=True, annot_kws={'size': 8})
        ax.set_title("Estimated noise transition matrix\n"+flip_type + '-' + noise_rate)

        plot_idx += 2
    plt.savefig(path_save)
    plt.show()


def plot_train_val_acc(path_save="../output/acc.png"):
    dist_cols = 4
    dist_rows = 3
    plt.figure(figsize=(6 * dist_cols, 5 * dist_rows))
    plot_idx = 1
    for dataset in ["mnist", "cifar10", "cifar100"]:
        for nose_rate in ["0.2", "0.5"]:
            path = dataset+"_symmetric_"+nose_rate+"_NoVol"
            res_df = pd.read_csv("../output/"+path+"/result.csv")

            ax = plt.subplot(dist_rows, dist_cols, plot_idx)
            ax = sns.lineplot(data=res_df[['train_acc', 'val_acc']])
            ax.set_xlabel("Epoch")
            plt.ylim((0, 1))  # fix y axis range
            ax.set_ylabel("Accuracy")
            ax.set_title(dataset.upper() + ', Symmetric-' + nose_rate)
            plot_idx += 1

        for nose_rate in ["0.2", "0.45"]:
            path = dataset+"_pair_"+nose_rate+"_NoVol"
            res_df = pd.read_csv("../output/" + path + "/result.csv")
            ax = plt.subplot(dist_rows, dist_cols, plot_idx)
            ax = sns.lineplot(data=res_df[['train_acc', 'val_acc']])
            ax.set_xlabel("Epoch")
            plt.ylim((0, 1))
            ax.set_ylabel("Accuracy")
            ax.set_title(dataset.upper()+', Pair-'+nose_rate)
            plot_idx += 1
    plt.savefig(path_save)
    plt.show()


def plot_train_val_loss(path_save="../output/loss.png"):
    dist_cols = 4
    dist_rows = 3
    plt.figure(figsize=(4 * dist_cols, 5 * dist_rows))
    plot_idx = 1
    for dataset in ["mnist", "cifar10", "cifar100"]:
        for nose_rate in ["0.2", "0.5"]:
            path = dataset+"_symmetric_"+nose_rate
            res_df = pd.read_csv("../output/"+path+"/result.csv")
            res_df['train_vol_loss'] = res_df['train_vol_loss'] * 0.0001
            res_df["CE"] = res_df['train_loss'] - res_df['train_vol_loss']

            ax = plt.subplot(dist_rows, dist_cols, plot_idx)
            ax = sns.lineplot(data=res_df[['CE', 'train_vol_loss']])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(dataset.upper() + ', Symmetric-' + nose_rate)
            ax = ax.legend(["CE", "Volume"])

            plot_idx += 1

        for nose_rate in ["0.2", "0.45"]:
            path = dataset+"_pair_"+nose_rate
            res_df = pd.read_csv("../output/" + path + "/result.csv")
            res_df['train_vol_loss'] = res_df['train_vol_loss'] * 0.0001
            res_df["CE"] = res_df['train_loss'] - res_df['train_vol_loss']

            ax = plt.subplot(dist_rows, dist_cols, plot_idx)
            ax = sns.lineplot(data=res_df[['CE', 'train_vol_loss']])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(dataset.upper() + ', Pair-' + nose_rate)
            ax = ax.legend(["CE", "Volume"])

            plot_idx += 1
    plt.savefig(path_save)
    plt.show()


if __name__ == "__main__":
    read_test_acc()
    transition_matrix_visual()
    plot_train_val_acc()
    plot_train_val_loss()
