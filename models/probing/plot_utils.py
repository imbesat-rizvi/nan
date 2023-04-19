import torch
from matplotlib import pyplot as plt
import seaborn as sns


def plot_reconstruction(x, x_pred, interp_range=None, save_path=None):
    x = torch.cat([i.dataset[:][0] for i in x])
    x_pred = torch.cat([torch.cat(i) for i in x_pred])

    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=x_pred, ax=ax)
    if interp_range:
        sns.lineplot(x=interp_range, y=interp_range, ax=ax, color="red")

    plt.grid()
    plt.xlabel("X")
    plt.ylabel("$\hat{X}$")
    plt.title("Reconstruction Plot")

    if save_path:
        fig.savefig(save_path)
    return fig
