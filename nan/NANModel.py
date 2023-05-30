import torch

from .NANEmbedder import NANEmbedder


class NANModel(torch.nn.Module):
    def __init__(
        self,
        net,  # model e.g. BERT which is wrapped by NANModel
        embedder=None,  # Embedder object in which case embedder_kwargs is ignored
        embedder_kwargs=dict(
            emb_net="fcn",
            emb_kwargs=dict(
                output_size=768,  # emb size of the upstream model e.g. BERT
                num_layers=1,
                dropout=0.2,  # immaterial for 1 layer NN
                non_linearity="ReLU",  # immaterial for 1 layer NN
            ),
            nums_kwargs=dict(
                digit_kwargs=dict(int_decimals=12, frac_decimals=7, scale=True)
            ),
            aux_kwargs={},  # default encode_aux kwargs
            use_aux=False,
            random_state=42,
        ),
    ):

        super().__init__()

        self.net = net
        self.embedder = embedder
        if embedder is None or embedder == "NANEmbedder":
            self.embedder = NANEmbedder(**embedder_kwargs)

    def forward(self, input_ids=None, nums=None, num_mask=None, **kwargs):
        output = dict()
        if input_ids is not None:
            net_emb = self.net.base_model.embeddings(input_ids=input_ids)
            if nums is not None:
                nan_emb = self.embedder(nums, num_mask=num_mask, all_emb=net_emb)
            output = self.net(inputs_embeds=nan_emb, **kwargs)
        return output
