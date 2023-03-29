import numpy as np
from torch.utils.data import DataLoader


from .utils import train_val_test_split


DATALOADER_KWARGS = dict(
    batch_size=64,
    shuffle=False,
    num_workers=1,
)


def gen_reconstruct_set(
    number_range=(-2000, 2000),
    interp_range=(-500, 500),
    as_int=True,
    train_size=0.6,
    test_size=0.2,
    random_state=42,
    dataloader_kwargs=DATALOADER_KWARGS,
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

    if dataloader_kwargs:
        train_data = DataLoader(train_data, **dataloader_kwargs)
        val_data = DataLoader(val_data, **dataloader_kwargs)
        test_data = DataLoader(test_data, **dataloader_kwargs)

    return train_data, val_data, test_data
