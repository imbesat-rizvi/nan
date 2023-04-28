import re
import torch
from transformers import BertTokenizer
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from .tokenizer_utils import split_and_tag_at_nums, tagged_tokens_to_nums


class NANBertTokenizer(BertTokenizer):
    EXP_TOK = "##e"

    def _tokenize(self, text):
        splits, is_num = split_and_tag_at_nums(text)

        def is_exp_tok(tok, pos):
            return is_num[pos - 1] and is_num[pos + 1] and tok in ("e", "E")

        split_tokens = []
        for i, (txt, is_n) in enumerate(zip(splits, is_num)):
            if is_n:
                split_tokens.append(txt)
            elif 0 < i < len(splits) - 1 and is_exp_tok(txt, i):
                split_tokens.append(self.EXP_TOK)
            else:
                split_tokens += super()._tokenize(txt)

        return split_tokens

    def _encode_plus(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        truncation_strategy=TruncationStrategy.DO_NOT_TRUNCATE,
        max_length=None,
        stride=0,
        is_split_into_words=False,
        pad_to_multiple_of=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        **kwargs,
    ):
        first_text_toks = self.tokenize(text, **kwargs)
        # first and last 0s for [CLS] and [SEP]
        text_nums = [0] + tagged_tokens_to_nums(first_text_toks) + [0]

        second_text_toks, second_text_nums = None, None
        if text_pair:
            second_text_toks = self.tokenize(text_pair, **kwargs)
            # last 0 for [SEP]
            text_nums += tagged_tokens_to_nums(second_text_toks) + [0]

        encoded = super()._encode_plus(
            text=first_text_toks,
            text_pair=second_text_toks,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

        if return_tensors:
            text_nums = torch.tensor(text_nums)
        encoded["nums"] = text_nums
        return encoded
