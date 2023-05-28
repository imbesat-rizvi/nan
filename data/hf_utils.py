import re
from functools import partial
from datasets import load_dataset, concatenate_datasets, DatasetDict


NUM_PAT = re.compile(r"([-+])?\d+(\.\d+)?")

FILTERS = dict(
    # TODO: support exponentials
    is_number=lambda x: re.fullmatch(NUM_PAT, x)
    is not None,
)

TRANSFORMS = dict(
    to_number=float,
)


def concat_datasets(dataset):
    if isinstance(dataset[0], DatasetDict):
        concat = DatasetDict()
        for split in dataset[0].keys():
            concat[split] = concatenate_datasets([d[split] for d in dataset])
    else:
        concat = concatenate_datasets(dataset)
    return concat


def get_dataset(path="allenai/lila", name="GSM8k_structured", split=None):
    if isinstance(name, str) or name is None:
        name = [name]
    dataset = [load_dataset(path, name=n, split=split) for n in name]
    dataset = dataset[0] if len(dataset) == 1 else concat_datasets(dataset)
    return dataset


def filter_dataset(dataset, col_filter=dict(output_answer="is_number")):
    def filter_func(ex, col, filt):
        filt = FILTERS.get(filt, filt)
        if isinstance(ex[col], list):
            filtered = [filt(i) for i in ex[col]]
        else:
            filtered = filt(ex[col])
        return filtered

    for col, filt in col_filter.items():
        f_func = partial(filter_func, col=col, filt=filt)
        dataset = dataset.filter(f_func, batched=True)
    return dataset


def transform_dataset(dataset, col_transform=dict(output_answer="to_number")):
    def transform_func(ex, col, transform):
        transform = TRANSFORMS.get(transform, transform)
        if isinstance(ex[col], list):
            ex[col] = [transform(i) for i in ex[col]]
        else:
            ex[col] = transform(ex[col])
        return ex

    for col, transform in col_transform.items():
        t_func = partial(transform_func, col=col, transform=transform)
        dataset = dataset.map(t_func, batched=True)
    return dataset


def check_and_resplit(dataset, test_split="validation", test_size=0.1, seed=42):
    if not (dataset.get(test_split) and len(dataset[test_split]) > 0):
        dataset["train"], dataset[test_split] = (
            dataset["train"].train_test_split(test_size=test_size, seed=seed).values()
        )
    return dataset


def get_processed_dataset(
    path="allenai/lila",
    name="GSM8k_structured",
    split=None,
    filter_kwargs=dict(col_filter=dict(output_answer="is_number")),
    transform_kwargs=dict(col_transform=dict(output_answer="to_number")),
    resplit_kwargs=dict(test_split="validation", test_size=0.1, seed=42),
):

    dataset = get_dataset(path, name=name, split=split)
    dataset = filter_dataset(dataset, **filter_kwargs)
    dataset = transform_dataset(dataset, **transform_kwargs)
    dataset = check_and_resplit(dataset, **resplit_kwargs)
    return dataset
