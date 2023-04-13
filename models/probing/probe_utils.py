import torch

from functools import partial

from nan.encoding import encode_nums, encode_aux


def get_embedder(emb_name="encoding", emb_args=dict(use_aux=True, nums={}, aux={})):
    if emb_name == "encoding":
        if emb_args.get("use_aux", False):
            num_encoder = partial(encode_nums, **emb_args["nums"])
            embedder = lambda x: torch.hstack(
                (
                    num_encoder(x),
                    encode_aux(x, num_encoder=num_encoder, **emb_args["aux"]),
                )
            )
        else:
            embedder = lambda x: encode_nums(x, **emb_args["nums"])

    return embedder
