from transformers.tokenization_utils_base import BatchEncoding

from .tokenizer_utils import split_and_tag_at_nums, tagged_tokens_to_nums


class NANTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.exp_tok = tokenizer.tokenize("1e3")[1]
        self.tokenizer.add_tokens([self.exp_tok], special_tokens=True)

    def tokenize(self, text, **kwargs):
        splits, is_num = split_and_tag_at_nums(text)

        def is_exp_tok(tok, pos):
            return is_num[pos - 1] and is_num[pos + 1] and tok in ("e", "E")

        tokens = []
        for i, (txt, is_n) in enumerate(zip(splits, is_num)):
            if is_n:
                tokens.append(txt)
            elif 0 < i < len(splits) - 1 and is_exp_tok(txt, i):
                tokens.append(self.exp_tok)
            else:
                tokens += self.tokenizer.tokenize(txt, **kwargs)

        return tokens

    def get_toks_nums_num_mask(self, text, **kwargs):
        toks = self.tokenize(text, **kwargs)
        nums, num_mask = tagged_tokens_to_nums(toks)
        toks = [self.tokenizer.unk_token if nm else t for t, nm in zip(toks, num_mask)]
        return toks, nums, num_mask

    def get_batched_toks_nums_num_mask(self, text, text_pair=None, **kwargs):
        tokenized_text, tokenized_text_pair = [], []
        num_pairs, num_mask_pairs = [], []

        if text_pair is None:
            text_pair = [None] * len(text)
            tokenized_text_pair = None

        for txt, txt_pair in zip(text, text_pair):
            txt_toks, txt_nums, txt_num_mask = self.get_toks_nums_num_mask(
                txt, **kwargs
            )

            pair_nums, pair_num_mask = None, None
            if txt_pair:
                pair_toks, pair_nums, pair_num_mask = self.get_toks_nums_num_mask(
                    txt_pair, **kwargs
                )
                tokenized_text_pair.append(pair_toks)

            tokenized_text.append(txt_toks)
            num_pairs.append((txt_nums, pair_nums))
            num_mask_pairs.append((txt_num_mask, pair_num_mask))

        return tokenized_text, tokenized_text_pair, num_pairs, num_mask_pairs

    def prepare_nums_outputs(
        self,
        num_pairs,
        num_mask_pairs,
        padding="max_length",
        truncation=True,
        max_length=None,
        pad_to_multiple_of=None,
    ):

        pad, trunc, max_len, _ = self.tokenizer._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        nums_outputs = {"nums": num_pairs, "num_mask": num_mask_pairs}
        for k, v in nums_outputs.items():
            nums_outputs[k] = self.tokenizer._batch_prepare_for_model(
                v,
                padding_strategy=pad,
                truncation_strategy=trunc,
                max_length=max_len,
                pad_to_multiple_of=pad_to_multiple_of,
            )["input_ids"]

        # rectify mask for special tokens such as CLS in num_mask
        nums_outputs["num_mask"] = [
            [int(i == 1) for i in mask] for mask in nums_outputs["num_mask"]
        ]

        return nums_outputs

    def __call__(
        self,
        text,
        text_pair=None,
        padding="max_length",
        truncation=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors="pt",
        **kwargs,
    ):
        if isinstance(text, str):
            text = [text]
            if isinstance(text_pair, str):
                text_pair = [text_pair]

        txt, txt_pair, num_pairs, num_mask_pairs = self.get_batched_toks_nums_num_mask(
            text, text_pair, **kwargs
        )

        outputs = self.tokenizer(
            txt,
            txt_pair,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            is_split_into_words=True,
            **kwargs,
        )

        nums_outputs = self.prepare_nums_outputs(
            num_pairs,
            num_mask_pairs,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        outputs["nums"] = nums_outputs.pop("nums")
        outputs["num_mask"] = nums_outputs.pop("num_mask")
        outputs = BatchEncoding(outputs, tensor_type=return_tensors)

        return outputs
