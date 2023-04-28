from nan import NANEncoder


def get_embedder(
    emb_name="encoding",
    emb_kwargs=dict(nums_kwargs={}, aux_kwargs={}, use_aux="also", random_state=42),
):
    if emb_name == "encoding":
        embedder = NANEncoder(**emb_kwargs).get_encoder()

    return embedder
