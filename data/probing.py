import numpy as np


from .utils import train_val_test_split


def gen_reconstruct_set(
    number_range=(-2000, 2000),
    interp_range=(-500, 500),
    as_int=True,
    train_size=0.6,
    test_size=0.2,
    random_state=42,
):

    val_size = 1 - train_size - test_size
    rng = np.random.default_rng(random_state)

    if as_int:
        nums = rng.permutation(np.arange(*(number_range)))
    else:
        size = number_range[1] - number_range[0]
        nums = rng.uniform(*number_range, size=size)

    if interp_range:
        interp_valid = np.logical_and(nums >= interp_range[0], nums <= interp_range[1])
        interp_nums = nums[interp_valid]
        exterp_nums = nums[~interp_valid]

        train_set, val_set, test_set = train_val_test_split(
            interp_nums,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )

        test_set = np.append(test_set, exterp_nums)

    else:
        train_set, val_set, test_set = train_val_test_split(
            nums,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )

    return train_set, val_set, test_set
