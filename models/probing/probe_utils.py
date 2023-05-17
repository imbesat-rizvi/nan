from nan import NANEmbedder


def get_embedder(
    emb_name="NANEmbedder",
    emb_kwargs=dict(
        emb_net=None, nums_kwargs={}, aux_kwargs={}, use_aux="also", random_state=42
    ),
):
    if emb_name == "NANEmbedder":
        embedder = NANEmbedder(**emb_kwargs)

    return embedder
