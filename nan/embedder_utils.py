import numpy as np
import torch


def scaled_range_func(x, scale_exp=13):
    x = x.unsqueeze(-1) * (10.0 ** torch.arange(-scale_exp + 1, scale_exp))
    x[torch.abs(x) < 0.1] = 0
    x[torch.abs(x) >= 1] = 0
    return x


def polar_to_NDim_cartesian(theta, dim=2):
    r"""From DICE Embedding paper:
    https://aclanthology.org/2020.emnlp-main.384/
    power of sins' dependent on precision of theta"""
    theta = theta.unsqueeze(-1)
    x = torch.sin(theta) ** torch.arange(0, dim)
    x[..., :-1] *= torch.cos(theta)
    return x


def arctan_func(x):
    return 2 * torch.arctan(x) / torch.pi


def periodic_func(x):
    return torch.stack((torch.sin(x), torch.cos(x)), dim=-1).view(*x.shape[:-1], -1)


def func_encoder(
    nums,
    func=arctan_func,
    scale_base=10,
    scale_exp=13,
    exp_divisor=1,
    symmetric_exp=True,
    skip_zero_exp=False,
    tol=1e-12,
):
    """
    Returns an encoded array obtained by applying the specified function to a
    scaled array as:

    encoded = func \left( \frac{nums}{ scale\_base^{ \frac{scale\_exp}{exp\_divisor} } } \right)

    with additional choices specified by other parameters.

    Parameters:
    -----------
    nums : array-like
        Input array to be encoded.

    func : function, optional (default=arctan_func)
        Function to be applied to the input array.

    scale_base : float, optional (default=10)
        Base value to be used for scaling the encoding.

    scale_exp : +ve int or array-like, optional (default=13)
        Exponential values to be used for scaling the encoding.

    exp_divisor : int, optional (default=1)
        Divisor to be used for dividing the exponent.
        Other schemes can have their respective values
        e.g. 768//2 in original sinusoidal embeddings.

    symmetric_exp : bool, optional (default=True)
        If True, extend exp to negative values too, symmetrically.

    skip_zero_exp : bool, optional (default=False)
        If True, skips the zero exp scaling i.e. the non-scaled number itself.

    tol : float, optional (default=1e-12)
        Tolerance value for rounding values close to zero.

    Returns:
    --------
    result : array-like
        Encoded array obtained by applying the specified function to the scaled input array.
    """

    # work on log scale and then exponentiate for correct precision
    log_scale_base = torch.log(torch.tensor(scale_base))

    exps = scale_exp
    if isinstance(scale_exp, int):
        exps = torch.arange(1, scale_exp)
        if not skip_zero_exp:
            exps = torch.cat((torch.tensor([0]), exps))
        if symmetric_exp:
            exps = torch.cat((torch.arange(-scale_exp + 1, 0), exps))

    scale = torch.exp(-exps * log_scale_base / exp_divisor)

    encoding = func(nums.unsqueeze(-1) * scale.to(nums.device))
    encoding[torch.abs(encoding) < tol] = 0
    return encoding


def digit_encoder(x, int_decimals=12, frac_decimals=12, scale=True):
    # precision of x affects precision of encoding
    def digits_from_pos_ints(n, decimals):
        decimals = torch.pow(10, torch.arange(decimals - 1, -1, -1))
        decimals = decimals.expand(*n.shape, len(decimals))

        digits = n.unsqueeze(-1) / decimals
        digits = torch.floor(torch.fmod(digits, 10))

        if scale:
            digits /= 10  # scale digits from 0 to 1
        return digits

    frac, intgr = np.modf(torch.abs(x).cpu())

    int_digits = digits_from_pos_ints(intgr, int_decimals)

    frac_digits = torch.floor(frac * 10**frac_decimals)
    can_be_safely_rounded = frac_digits < 10**frac_decimals - 1  # i.e. not all 9s
    need_to_be_rounded = torch.fmod(frac_digits, 100) == 99  # i.e. last 2 digits 9
    frac_digits += can_be_safely_rounded * need_to_be_rounded
    frac_digits = digits_from_pos_ints(frac_digits, frac_decimals)

    sign = torch.sign(x).unsqueeze(-1)
    encoding = torch.cat((sign, int_digits.to(x.device), frac_digits.to(x.device)), -1)
    return encoding


def digit_decoder(encoding, int_decimals=12, scaled=True, with_sign=True):
    encoding = torch.atleast_2d(encoding)
    digit_start_pos = 1 if with_sign else 0

    assert encoding.shape[-1] >= int_decimals + digit_start_pos

    frac_decimals = encoding.shape[-1] - int_decimals - digit_start_pos
    if scaled:
        encoding[..., digit_start_pos:] *= 10

    decimals = torch.pow(10.0, torch.arange(int_decimals - 1, -frac_decimals - 1, -1))
    decoding = (encoding[..., digit_start_pos:] * decimals).sum(dim=-1)
    if with_sign:
        decoding *= encoding[..., 0]

    return decoding


def order_encoder(x, scale_exp=13):
    return func_encoder(x, func=arctan_func, scale_exp=scale_exp)
    # return scaled_range_func(x, scale_exp=scale_exp)


def sinusoidal_encoder(x, scale_base=10_000, exp_divisor=50):

    encoding = func_encoder(
        x,
        func=periodic_func,
        scale_base=scale_base,
        scale_exp=exp_divisor,
        exp_divisor=exp_divisor,
        symmetric_exp=False,
    )

    return encoding


def dice_encoder(x, low=0, high=1000, dim=10, Q=None, random_state=42):
    r"""From DICE Embedding paper:
    https://aclanthology.org/2020.emnlp-main.384/"""
    if Q is None:
        rng = torch.Generator().manual_seed(random_state)
        M = torch.normal(mean=0, std=1, size=(dim, dim), generator=rng)
        Q, _ = torch.linalg.qr(M, mode="complete")

    assert high > low

    # clip values so that theta lies between 0 and pi
    # this is different than random mapping in the source paper
    x = torch.clip(x, min=low, max=high)
    theta = (x - low) * torch.pi / (high - low)
    v = polar_to_NDim_cartesian(theta, dim=dim)
    encoding = torch.matmul(Q, v.mT).mT
    return encoding


def encode_nums(
    nums,
    digit_kwargs=dict(int_decimals=12, frac_decimals=12, scale=True),
    order_kwargs=dict(scale_exp=13),
    sinusoidal_kwargs=dict(scale_base=10000, exp_divisor=50),
    dice_kwargs=dict(low=0, high=1000, dim=10, Q=None, random_state=42),
    concat=True,
):

    encoding = []
    if digit_kwargs:
        encoding.append(digit_encoder(nums, **digit_kwargs))
    if order_kwargs:
        encoding.append(order_encoder(nums, **order_kwargs))
    if sinusoidal_kwargs:
        encoding.append(sinusoidal_encoder(nums, **sinusoidal_kwargs))
    if dice_kwargs:
        encoding.append(dice_encoder(nums, **dice_kwargs))

    if concat:
        encoding = torch.cat(encoding, dim=-1)
    return encoding


def log_func(x):
    eps = (x == 0) * torch.finfo(torch.get_default_dtype()).eps
    return torch.log(torch.abs(x) + eps)


def encode_aux(nums, num_encoder=encode_nums, log_aux=True, concat=True):
    # concat of num_encoder should be True when using encode_aux
    aux_encoding = []
    if log_aux:
        aux_encoding.append(num_encoder(log_func(nums)))

    if concat:
        aux_encoding = torch.cat(aux_encoding, dim=-1)
    return aux_encoding
