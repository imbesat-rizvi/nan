from sklearn.model_selection import train_test_split


def train_val_test_split(
    data,
    train_size=0.6,
    test_size=0.2,
    random_state=42,
    **kwargs,
):

    test_size = min(test_size, 1 - train_size)
    val_size = 1 - train_size - test_size

    train_set, val_test_set = train_test_split(
        data,
        train_size=train_size,
        random_state=random_state,
        **kwargs,
    )

    val_set, test_set = train_test_split(
        val_test_set,
        train_size=val_size / (val_size + test_size),
        random_state=random_state,
        **kwargs,
    )

    return train_set, val_set, test_set
