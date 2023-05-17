import torch

from .NANEmbedder import NANEmbedder


class NANModel(torch.nn.Module):
    def __init__(
        self,
        net,  # model e.g. BERT which is wrapped by NANModel
        embedder=None,  # NANEmbedder in which case embedder_kwargs is ignored
        embedder_kwargs=dict(
            emb_net="fcn",
            emb_kwargs=dict(
                output_size=768,  # emb size of the upstream model e.g. BERT
                num_layers=3,
                dropout=0.2,
                non_linearity="ReLU",
            ),
            nums_kwargs={},  # default encode_nums kwargs
            aux_kwargs={},  # default encode_aux kwargs
            use_aux="also",
            random_state=42,
        ),
    ):

        super().__init__()

        self.net = net
        self.embedder = embedder
        if embedder is None:
            self.embedder = NANEmbedder(**embedder_kwargs)

    def forward(self, input_ids, nums, num_mask, **kwargs):
        net_emb = self.net.base_model.embeddings(input_ids=input_ids)
        nan_emb = self.embedder(nums, num_mask=num_mask, all_emb=net_emb)
        output = self.net(inputs_embeds=nan_emb, **kwargs)
        return output
