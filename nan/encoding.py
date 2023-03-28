import numpy as np


def arctan_func(x):
    return 2 * np.arctan(x) / np.pi


def periodic_func(x):
    return np.dstack((np.sin(x), np.cos(x))).reshape(x.shape[0], -1)


def func_encoder(
    num,
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

    encoded = func \left( \frac{num}{ scale\_base^{ \frac{scale\_exp}{exp\_divisor} } } \right)

    with additional choices specified by other parameters.

    Parameters:
    -----------
    num : array-like
        Input array to be encoded.

    func : function, optional (default=arctan_func)
        Function to be applied to the input array.

    scale_base : float, optional (default=10)
        Base value to be used for scaling the encoding.

    scale_exp : +ve int or array-like, optional (default=13)
        Exponential values to be used for scaling the encoding.

    exp_step : int, optional (default=1)
        step value for arange when scale_exp array is to be generated
        based on its provided int value.

    exp_divisor : int, optional (default=1)
        Divisor to be used for dividing the exponent.
        Other schemes can have their respective values
        e.g. 768 in original sinusoidal embeddings.

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
    log_scale_base = np.log(scale_base)

    exps = scale_exp
    if isinstance(scale_exp, int):
        exps = np.arange(1, scale_exp)
        if not skip_zero_exp:
            exps = np.concatenate(([0], exps))
        if symmetric_exp:
            exps = np.concatenate((np.arange(-scale_exp + 1, 0), exps))

    scale = np.exp(-exps * log_scale_base / exp_divisor)

    num = np.atleast_1d(num).reshape((-1, 1))
    encoding = func(num * scale)

    encoding[np.abs(encoding) < tol] = 0
    return encoding


def digit_encoder(x, int_decimals=12, frac_decimals=12):
    def digits_from_pos_ints(n, decimals):
        digits = n.reshape(-1, 1) / np.power(10, np.arange(decimals - 1, -1, -1))
        digits = np.floor(np.mod(digits, 10)) / 10  # scale digits from 0 to 1
        return digits

    frac, intgr = np.modf(np.abs(x))

    int_digits = digits_from_pos_ints(intgr, int_decimals)

    frac_digits = np.floor(frac * 10**frac_decimals)
    can_be_safely_rounded = frac_digits < 10**frac_decimals - 1  # i.e. not all 9s
    need_to_be_rounded = np.mod(frac_digits, 100) == 99  # i.e. last 2 digits 9
    frac_digits += can_be_safely_rounded * need_to_be_rounded
    frac_digits = digits_from_pos_ints(frac_digits, frac_decimals)

    sign = np.sign(x).reshape(-1, 1)
    encoding = np.hstack((sign, int_digits, frac_digits))
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


def encode_num(
    num,
    digit_kwargs=dict(int_decimals=12, frac_decimals=12),
    order_kwargs=dict(scale_exp=13),
    sinusoidal_kwargs=dict(scale_base=10000, exp_divisor=50),
):

    encoding = np.array([])
    if not isinstance(num, (int, float)):
        encoding = np.empty(shape=(len(num), 0))

    if digit_kwargs:
        encoding = np.hstack((encoding, digit_encoder(num, **digit_kwargs)))
    if order_kwargs:
        encoding = np.hstack((encoding, order_encoder(num, **order_kwargs)))
    if sinusoidal_kwargs:
        encoding = np.hstack((encoding, sinusoidal_encoder(num, **sinusoidal_kwargs)))

    return encoding


def log_func(x):
    eps = (x == 0) * np.finfo("float").eps
    return np.log(np.abs(x) + eps)


def encode_aux(num, num_encoder=encode_num, log_aux=True):

    aux_encoding = np.array([])
    if not isinstance(num, (int, float)):
        aux_encoding = np.empty(shape=(len(num), 0))

    if log_aux:
        aux_encoding = np.hstack((aux_encoding, num_encoder(log_func(num))))

    return aux_encoding
