import torch
from matplotlib import pyplot as plt
import seaborn as sns


def plot_reconstruction(x, x_pred, save_path=None):
    x = torch.cat([i.dataset for i in x])
    x_pred = torch.cat([torch.cat(i) for i in x_pred])

    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=x_pred, ax=ax)

    plt.grid()
    plt.xlabel("X")
    plt.ylabel("$\hat{X}$")
    plt.title("Reconstruction Plot")

    if save_path:
        fig.savefig(save_path)
    return fig