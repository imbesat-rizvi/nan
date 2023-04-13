import torch
from torch import nn


def create_fcn(
    in_size,
    num_outputs=2,
    num_layers=2,
    hidden_size=64,
    lin_kwargs={},
    dropout=0.2,
    non_linearity="ReLU",
):

    fcn = []
    for i in range(num_layers):
        in_size = hidden_size if i > 0 else in_size
        out_size = hidden_size if i < num_layers - 1 else num_outputs

        fcn.append(nn.Linear(in_features=in_size, out_features=out_size, **lin_kwargs))

        if i < num_layers - 1:
            fcn.append(getattr(nn, non_linearity)())
            if dropout:
                fcn.append(nn.Dropout(p=dropout))

    return nn.Sequential(*fcn)


def create_fc_task_nets(
    in_size,
    num_tasks=2,
    task_out_sizes=1,
    head_kwargs=dict(
        num_layers=2,
        hidden_size=64,
        dropout=0.2,
        non_linearity="ReLU",
    ),
):

    fc_task_nets = nn.ModuleList()
    for i in range(num_tasks):

        fcn_kwargs = head_kwargs
        if not isinstance(head_kwargs, dict):
            fcn_kwargs = head_kwargs[i]

        fcn_outputs = task_out_sizes
        if not isinstance(task_out_sizes, int):
            fcn_outputs = task_out_sizes[i]

        task_net = create_fcn(
            in_size=in_size,
            num_outputs=fcn_outputs,
            **fcn_kwargs,
        )

        fc_task_nets.append(task_net)

    return fc_task_nets
