import numpy as np
import torch


def arctan_func(x):
    return 2 * torch.arctan(x) / torch.pi


def periodic_func(x):
    return torch.dstack((torch.sin(x), torch.cos(x))).reshape(x.shape[0], -1)


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
    log_scale_base = torch.log(scale_base)

    exps = scale_exp
    if isinstance(scale_exp, int):
        exps = torch.arange(1, scale_exp)
        if not skip_zero_exp:
            exps = torch.cat((torch.tensor([0]), exps))
        if symmetric_exp:
            exps = torch.cat((torch.arange(-scale_exp + 1, 0), exps))

    scale = torch.exp(-exps * log_scale_base / exp_divisor)

    nums = torch.atleast_1d(nums).reshape((-1, 1))
    encoding = func(nums * scale)

    encoding[torch.abs(encoding) < tol] = 0
    return encoding


def digit_encoder(x, int_decimals=12, frac_decimals=12):
    def digits_from_pos_ints(n, decimals):
        digits = n.reshape(-1, 1) / torch.pow(10, torch.arange(decimals - 1, -1, -1))
        digits = torch.floor(torch.fmod(digits, 10)) / 10  # scale digits from 0 to 1
        return digits

    frac, intgr = np.modf(torch.abs(x).cpu())

    int_digits = digits_from_pos_ints(intgr, int_decimals)

    frac_digits = torch.floor(frac * 10**frac_decimals)
    can_be_safely_rounded = frac_digits < 10**frac_decimals - 1  # i.e. not all 9s
    need_to_be_rounded = torch.fmod(frac_digits, 100) == 99  # i.e. last 2 digits 9
    frac_digits += can_be_safely_rounded * need_to_be_rounded
    frac_digits = digits_from_pos_ints(frac_digits, frac_decimals)

    sign = torch.sign(x).reshape(-1, 1)
    encoding = torch.hstack((sign, int_digits, frac_digits))
    return encoding


def order_encoder(x, scale_exp=13):
    return func_encoder(x, func=arctan_func, scale_exp=scale_exp)


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


def encode_nums(
    nums,
    digit_kwargs=dict(int_decimals=12, frac_decimals=12),
    order_kwargs=dict(scale_exp=13),
    sinusoidal_kwargs=dict(scale_base=10000, exp_divisor=50),
):

    num_counts = len(nums) if not isinstance(nums, (int, float)) else 1
    encoding = torch.empty(size=(num_counts, 0))

    if digit_kwargs:
        encoding = torch.hstack((encoding, digit_encoder(nums, **digit_kwargs)))
    if order_kwargs:
        encoding = torch.hstack((encoding, order_encoder(nums, **order_kwargs)))
    if sinusoidal_kwargs:
        encoding = torch.hstack((encoding, sinusoidal_encoder(nums, **sinusoidal_kwargs)))

    return encoding if num_counts > 1 else encoding[0]


def log_func(x):
    eps = (x == 0) * torch.finfo(torch.get_default_dtype()).eps
    return torch.log(torch.abs(x) + eps)


def encode_aux(nums, num_encoder=encode_nums, log_aux=True):

    num_counts = len(nums) if not isinstance(nums, (int, float)) else 1
    aux_encoding = torch.empty(size=(num_counts, 0))
    if log_aux:
        encoding = num_encoder(log_func(nums)).reshape(num_counts, -1)
        aux_encoding = torch.hstack((aux_encoding, encoding))

    return aux_encoding if num_counts > 1 else aux_encoding[0]
