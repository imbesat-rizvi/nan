import torch


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

        fcn.append(
            torch.nn.Linear(in_features=in_size, out_features=out_size, **lin_kwargs)
        )

        if i < num_layers - 1:
            fcn.append(getattr(torch.nn, non_linearity)())
            if dropout:
                fcn.append(torch.nn.Dropout(p=dropout))

    return torch.nn.Sequential(*fcn)
