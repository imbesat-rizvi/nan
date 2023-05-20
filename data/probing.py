import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


from .utils import train_val_test_split


DATALOADER_KWARGS = dict(
    batch_size=64,
    shuffle=False,
    num_workers=1,
)


def gen_rand_num_set(
    number_range=(-2000, 2000),
    interp_range=(-500, 500),
    as_int=True,
    num_floats=None,
    train_size=0.6,
    test_size=0.2,
    random_state=42,
):

    val_size = 1 - train_size - test_size
    rng = np.random.default_rng(random_state)

    if as_int:
        nums = rng.permutation(np.arange(*(number_range)))
    else:
        size = num_floats if num_floats else number_range[1] - number_range[0]
        nums = rng.uniform(*number_range, size=size)

    if interp_range:
        interp_valid = np.logical_and(nums >= interp_range[0], nums <= interp_range[1])
        interp_nums = nums[interp_valid]
        exterp_nums = nums[~interp_valid]

        train_data, val_data, test_data = train_val_test_split(
            interp_nums,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )

        test_data = np.append(test_data, exterp_nums)

    else:
        train_data, val_data, test_data = train_val_test_split(
            nums,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )

    return train_data, val_data, test_data


def gen_reconstruct_set(
    number_range=(-2000, 2000),
    interp_range=(-500, 500),
    as_int=True,
    num_floats=None,
    train_size=0.6,
    test_size=0.2,
    random_state=42,
    dataloader_kwargs=DATALOADER_KWARGS,
):

    train_data, val_data, test_data = gen_rand_num_set(
        number_range=number_range,
        interp_range=interp_range,
        as_int=as_int,
        num_floats=num_floats,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
    )

    if dataloader_kwargs:
        data = {"train": train_data, "val": val_data, "test": test_data}
        for k, d in data.items():
            d = torch.tensor(d, dtype=torch.get_default_dtype())
            data[k] = DataLoader(TensorDataset(d, d), **dataloader_kwargs)

        train_data, val_data, test_data = data.values()
    return train_data, val_data, test_data


def gen_arith_op_set(
    number_range=(-2000, 2000),
    interp_range=(-500, 500),
    ops=("add", "sub", "abs_diff", "mul", "div", "max", "argmax"),
    as_int=True,
    num_floats=None,
    train_size=0.6,
    test_size=0.2,
    random_state=42,
    dataloader_kwargs=DATALOADER_KWARGS,
):
    r"""Sythetic generation of two numbers and their result from arithmetic operations"""

    train_data, val_data, test_data = None, None, None
    rng = np.random.default_rng(random_state)
    random_state = rng.integers(low=0, high=100, size=2)

    for i in range(2):
        train, val, test = [
            np.atleast_2d(x).T
            for x in gen_rand_num_set(
                number_range=number_range,
                interp_range=interp_range,
                as_int=as_int,
                num_floats=num_floats,
                train_size=train_size,
                test_size=test_size,
                random_state=random_state[i],
            )
        ]

        if i == 0:
            train_data, val_data, test_data = train, val, test
        else:
            train_data = np.hstack((train_data, train))
            val_data = np.hstack((val_data, val))
            test_data = np.hstack((test_data, test))

    if isinstance(ops, str):
        ops = (ops,)

    ops_func = {
        "add": (lambda x, y: x + y),
        "sub": (lambda x, y: x - y),
        "abs_diff": (lambda x, y: abs(x - y)),
        "mul": (lambda x, y: x * y),
        "div": (lambda x, y: x / y),
        "max": (lambda x, y: np.maximum(x, y)),
        "argmax": (lambda x, y: np.vstack((x, y)).T.argmax(axis=1)),
    }

    data = {"train": train_data, "val": val_data, "test": test_data}
    for op in ops:
        for k, d in data.items():
            res = ops_func[op](d[:, 0], d[:, 1])
            data[k] = np.hstack((d, np.atleast_2d(res).T))

    train_data, val_data, test_data = data.values()

    if dataloader_kwargs:
        data = {"train": train_data, "val": val_data, "test": test_data}
        for k, d in data.items():
            d = torch.tensor(d, dtype=torch.get_default_dtype())
            data[k] = DataLoader(TensorDataset(d[:, :2], d[:, 2:]), **dataloader_kwargs)

        train_data, val_data, test_data = data.values()
    return train_data, val_data, test_data
