import re
import torch
from transformers import BertTokenizer
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy, BatchEncoding

from .tokenizer_utils import split_and_tag_at_nums, tagged_tokens_to_nums

# Tokenizer class adapting the base class with required functions modified
# Thus code heavily lifted from base class wherever possible


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

    def _get_toks_nums_num_mask(self, text, mask_for_cls=True, **tok_kwargs):
        toks = self.tokenize(text, **tok_kwargs)
        nums, num_mask = tagged_tokens_to_nums(toks)
        # 0s for [CLS] and [SEP]
        nums += [0]  # for [SEP]
        num_mask += [0]  # for [SEP]
        if mask_for_cls:
            nums = [0] + nums
            num_mask = [0] + num_mask
        return toks, nums, num_mask

    def _prepare_encoded_outputs_with_nums(
        self,
        nums,
        num_mask,
        encoded_batch=None,
        padding_strategy=PaddingStrategy.DO_NOT_PAD,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
    ):
        # alias nums as input_ids and num_mask as token_type_ids
        # so that self.pad can work on these
        nums_outputs = self.pad(
            {"input_ids": nums, "token_type_ids": num_mask},
            padding=padding_strategy.value,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=False,
        )

        if encoded_batch:
            encoded_batch["nums"] = nums_outputs["input_ids"]
            encoded_batch["num_mask"] = nums_outputs["token_type_ids"]
            encoded_batch = BatchEncoding(encoded_batch, tensor_type=return_tensors)
        else:
            nums_outputs["nums"] = nums_outputs.pop("input_ids")
            nums_outputs["num_mask"] = nums_outputs.pop("token_type_ids")
            encoded_batch = BatchEncoding(nums_outputs, tensor_type=return_tensors)

        return encoded_batch

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

        text_toks, nums, num_mask = self._get_toks_nums_num_mask(text, **kwargs)

        text_pair_toks = None
        if text_pair:
            text_pair_toks, n, nm = self._get_toks_nums_num_mask(
                text_pair, mask_for_cls=False, **kwargs
            )
            nums += n
            num_mask += nm

        outputs = super()._encode_plus(
            text=text_toks,
            text_pair=text_pair_toks,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=None,  # We convert the whole batch to tensors at the end
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

        outputs = self._prepare_encoded_outputs_with_nums(
            nums,
            num_mask,
            encoded_batch=outputs,
            padding_strategy=padding_strategy,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return outputs

    def _batch_encode_plus(
        self,
        batch_text_or_text_pairs,
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

        tokenized_text_or_text_pairs = []
        nums, num_mask = [], []

        for text_or_text_pair in batch_text_or_text_pairs:

            if not isinstance(text_or_text_pair, (list, tuple)):
                txt, txt_pair = text_or_text_pair, None
            elif is_split_into_words and not isinstance(
                text_or_text_pair[0], (list, tuple)
            ):
                txt, txt_pair = text_or_text_pair, None
            else:
                txt, txt_pair = text_or_text_pair

            txt_toks, txt_nums, txt_num_mask = self._get_toks_nums_num_mask(
                txt, **kwargs
            )

            txt_pair_toks = None
            txt_pair_nums, txt_pair_num_mask = [], []
            if txt_pair:
                (
                    txt_pair_toks,
                    txt_pair_nums,
                    txt_pair_num_mask,
                ) = self._get_toks_nums_num_mask(txt_pair, mask_for_cls=False, **kwargs)

            tokenized_text_or_text_pairs.append((txt_toks, txt_pair_toks))
            nums.append(txt_nums + txt_pair_nums)
            num_mask.append(txt_num_mask + txt_pair_num_mask)

        batch_outputs = super()._batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            add_special_tokens=add_special_tokens,
            padding_strategy=padding_strategy,
            truncation_strategy=truncation_strategy,
            max_length=max_length,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=None,  # We convert the whole batch to tensors at the end
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            **kwargs,
        )

        batch_outputs = self._prepare_encoded_outputs_with_nums(
            nums,
            num_mask,
            encoded_batch=batch_outputs,
            padding_strategy=padding_strategy,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return batch_outputs
