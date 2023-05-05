import re


unitary_look_behind = r"(?<=[\s\({\[eE])"  # space, parenth or exp
unitary_sign = f"({unitary_look_behind}[-+])?"  # max 1 preceding sign
# max comma grouping of 3; if decimal then at least 1 digit present after it
comma_based_number = r"\d{1,3}(,\d{3})*(\.\d+)?"
# Due to above, non-comma based numbers are split at a grouping of 3 digits
# Such splits are identified and combined as the split indices are shared
# in re.Match object.
# e.g. 10,251,515.987 identified as 10,251,515.987
# but 10251515.987 gets split as 102 515 15.987
NUMBER_PATTERN = re.compile(f"{unitary_sign}{comma_based_number}")


def combine_match_spans(m1, m2):
    r"""combine spans with common index e.g. (31,33), (33-37) -> (31,37)
    Required for numbers without commas in them but were split because of
    comma based number regex"""
    s1 = m1.span() if isinstance(m1, re.Match) else m1
    s2 = m2.span() if isinstance(m2, re.Match) else m2
    spans = [(s1[0], s2[1])] if s2[0] == s1[1] else [s1, s2]
    return spans


def split_and_tag_at_nums(text, tag="<N>"):
    number_matches = [x for x in re.finditer(NUMBER_PATTERN, text)]
    number_spans = [number_matches[0].span()] if number_matches else []
    for i in range(1, len(number_matches)):
        number_spans.extend(combine_match_spans(number_spans.pop(), number_matches[i]))

    splits, is_num = [], []
    prev_span_start = 0
    for span in number_spans:
        prev_span_end = span[0] if span[0] else 0
        txt = text[prev_span_start:prev_span_end]

        if txt:
            splits.append(txt.strip())
            is_num.append(0)

        txt = f"{tag}{text[span[0]:span[1]].replace(',','')}{tag}"
        splits.append(txt)
        is_num.append(1)
        prev_span_start = span[1]

    if prev_span_start != len(text):
        splits.append(text[prev_span_start:])
        is_num.append(0)

    return splits, is_num


def tagged_tokens_to_nums(tokens, tag="<N>"):
    def tok_to_num(txt):
        # is_num to differentiate between 0 as num vs 0 for text
        if txt[: len(tag)] == tag and txt[-len(tag) :] == tag:
            return float(txt[len(tag) : -len(tag)]), 1  # num, is_num
        return 0, 0  # num, is_num

    nums, is_num = zip(*(tok_to_num(i) for i in tokens))
    return list(nums), list(is_num)
