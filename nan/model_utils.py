import torch
from torch import nn

from .embedder_utils import digit_encoder, digit_decoder


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
    # task heads as module list to loop through to convert:
    # BxF -> BxHxF
    # B: Batch, H: Head, F: Feature

    fc_task_nets = []
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

    fc_task_nets = fc_task_nets[0] if num_tasks==1 else nn.ModuleList(fc_task_nets)
    return fc_task_nets


def digit_loss(inp, target, int_decimals=12):
    r"""inp: BxCxP or Bx(CxP), target: B or BxP
    B: Batch, P: Position of Digit, C: Class of digit
    P >= int_decimals, C in 0-9"""

    inp = torch.atleast_2d(inp)
    target = torch.atleast_1d(target)

    if len(inp.shape) == 2:
        # Bx(CxP) -> BxCxP
        inp = inp.view(inp.shape[0], 10, -1)

    assert inp.shape[-1] >= int_decimals

    if len(target.shape) == 1:
        frac_decimals = inp.shape[-1] - int_decimals
        # target: B -> BxP
        target = digit_encoder(target, int_decimals, frac_decimals, scale=False)
        # remove polarity info, convert to long
        target = target[..., 1:].to(torch.long)

    loss = nn.functional.cross_entropy(inp, target)
    return loss


def polarity_loss(inp, target):
    r"""inp: B, target: B, target is either nums or torch.sign(nums)"""
    polarity = (target >= 0).to(inp)
    loss = nn.functional.binary_cross_entropy_with_logits(inp, polarity)
    return loss


def digit_polarity_loss(inp, target, int_decimals=12):
    r"""inp: Bx(1+CxP), target: B or Bx(1+P)
    first entry (1) is for polarity, P >= int_decimals, C in 0-9"""

    inp = torch.atleast_2d(inp)
    target = torch.atleast_1d(target)

    digit_target = polarity_target = target
    if len(target.shape) == 2:
        digit_target = target[:, 1:]
        polarity_target = target[:, 0]

    loss = digit_loss(inp[:, 1:], digit_target, int_decimals)
    loss += polarity_loss(inp[:, 0], polarity_target)
    return loss


def digit_polarity_decoder(inp, int_decimals=12, normalized=False):
    r"""inp: Bx(1+CxP)
    first entry (1) is for polarity, P >= int_decimals, C in 0-9"""

    inp = torch.atleast_2d(inp)

    sign_thresh = 0.5 if normalized else 0
    sign = inp[:, 0] >= sign_thresh
    sign = 2 * sign - 1  # convert 0 1 to sign i.e. -1 1

    # Bx(1+CxP) -> BxCxP
    digits = inp[:, 1:].view(inp.shape[0], 10, -1)
    digits = digits.argmax(dim=1)  # digit from max value at digit pos

    decoding = digit_decoder(digits, int_decimals, scaled=False, with_sign=False)
    decoding *= sign
    return decoding
