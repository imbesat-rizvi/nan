import torch
from copy import deepcopy
from functools import partial

from .encoder_utils import encode_nums, encode_aux


class NANEncoder:
    def __init__(
        self,
        nums_kwargs={},  # default encode_nums kwargs
        aux_kwargs={},  # default encode_aux kwargs
        use_aux="also",
        random_state=42,
    ):

        assert use_aux in (False, "also", "only")

        self.nums_kwargs = deepcopy(nums_kwargs)
        self.aux_kwargs = deepcopy(aux_kwargs)
        self.use_aux = use_aux
        self.random_state = random_state

        if (
            self.nums_kwargs.get("dice_kwargs")
            and self.nums_kwargs["dice_kwargs"].get("Q") is None
        ):
            dim = self.nums_kwargs["dice_kwargs"].get("dim", 10)
            rng = torch.Generator().manual_seed(random_state)
            M = torch.normal(mean=0, std=1, size=(dim, dim), generator=rng)
            self.nums_kwargs["dice_kwargs"]["Q"], _ = torch.linalg.qr(
                M, mode="complete"
            )

        self.num_encoder = partial(encode_nums, **self.nums_kwargs)
        self.aux_encoder = partial(
            encode_aux, num_encoder=self.num_encoder, **self.aux_kwargs
        )

    def get_encoder(self):
        def combined_encoder(x):
            return torch.hstack((self.num_encoder(x), self.aux_encoder(x)))

        if not self.use_aux:
            encoder = self.num_encoder
        elif self.use_aux == "only":
            encoder = self.aux_encoder
        else:
            encoder = combined_encoder

        return encoder
