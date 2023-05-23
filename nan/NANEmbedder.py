import torch
from copy import deepcopy
from functools import partial

from .model_utils import create_fcn
from .embedder_utils import encode_nums, encode_aux


class NANEmbedder(torch.nn.Module):
    def __init__(
        self,
        emb_net="fcn",
        emb_kwargs=dict(
            output_size=768,  # emb size of the upstream model e.g. BERT
            num_layers=1,
            dropout=0.2,
            non_linearity="ReLU",
        ),
        nums_kwargs={},  # default encode_nums kwargs
        aux_kwargs={},  # default encode_aux kwargs
        use_aux="also",
        random_state=42,
    ):

        super().__init__()
        assert use_aux in (False, "also", "only")

        self.nums_kwargs = deepcopy(nums_kwargs)
        self.aux_kwargs = deepcopy(aux_kwargs)
        self.use_aux = use_aux
        self.random_state = random_state

        self.encoder = self.num_encoder = self.aux_encoder = None
        self.configure_encoder()

        self.net = lambda x: x
        if emb_net == "fcn":
            self.net = self._create_fcn_net(**emb_kwargs)

    def _init_num_encoder_states(self):

        if (
            self.nums_kwargs.get("dice_kwargs")
            and self.nums_kwargs["dice_kwargs"].get("Q") is None
        ):
            dim = self.nums_kwargs["dice_kwargs"].get("dim", 10)
            rng = torch.Generator().manual_seed(self.random_state)
            M = torch.normal(mean=0, std=1, size=(dim, dim), generator=rng)
            self.nums_kwargs["dice_kwargs"]["Q"], _ = torch.linalg.qr(
                M, mode="complete"
            )

    def configure_encoder(self):

        self._init_num_encoder_states()
        self.num_encoder = partial(encode_nums, **self.nums_kwargs)
        self.aux_encoder = partial(
            encode_aux, num_encoder=self.num_encoder, **self.aux_kwargs
        )

        def combined_encoder(x):
            return torch.cat((self.num_encoder(x), self.aux_encoder(x)), dim=-1)

        if not self.use_aux:
            self.encoder = self.num_encoder
        elif self.use_aux == "only":
            self.encoder = self.aux_encoder
        else:
            self.encoder = combined_encoder

    def _create_fcn_net(self, **emb_kwargs):
        in_size = self.encoder(torch.tensor([0])).shape[-1]
        net = create_fcn(
            in_size=in_size,
            num_outputs=emb_kwargs.get("output_size", 768),
            num_layers=emb_kwargs.get("num_layers", 3),
            hidden_size=emb_kwargs.get("output_size", 768) // 2,
            dropout=emb_kwargs.get("dropout", 0.2),
            non_linearity=emb_kwargs.get("non_linearity", "ReLU"),
        )
        return net

    def forward(self, nums, num_mask=None, all_emb=None):
        emb = self.net(self.encoder(nums))
        if all_emb is not None and num_mask is not None:
            # all_emb, num_emb are BxTxF while num_mask is BxT
            num_mask = num_mask.unsqueeze(-1)
            emb = all_emb * (1 - num_mask) + emb * num_mask
        return emb
