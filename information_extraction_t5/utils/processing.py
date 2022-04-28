"""Utility methods for pre and post processing."""
from collections import OrderedDict
from typing import Dict, List, Tuple

import regex as re


def get_intersection_set(list_a: List, list_b: List) -> set:
    """Returns the intersection set of two lists."""
    set_a = set(list_a)
    set_b = set(list_b)
    intersection = set_a.intersection(set_b)

    return intersection


def concat_or_terms(terms, suffix='{e<=1}'):
    """Concats a list of terms in an OR regex.

    Example:
    >>> concat_or_terms([r'foo', r'bar'], suffix='{e<=1}')
    '(?:foo|bar){e<=1}'

    Args:
        terms (list): terms to be considered in a regex search group
        suffix (str): fuzzy options to use in the search

    Returns:
        (str): regex string for group search

    """
    groups = '|'.join(map(str, terms))

    return r'(?:{}){}'.format(groups, suffix)


def expand_composite_char_pattern(text: str) -> str:
    """ Replace composable char in the given text for a regex group with all
    its composite versions.

    Args:
        text: the string to be expanded

    Returns:
        a new string with every composable char replaced by its composites
        pattern
    """

    composite_char_groups = [
        'aáàâã',
        'eéê',
        'ií',
        'oóõ',
        'uúü',
        'cç'
    ]

    for group in composite_char_groups:
        text = re.sub(fr'[{group}]', f'[{group}]', text)
    return text


def count_k_v(d):
    """Count keys and values in nested dictionary."""
    keys, values = 0, 0
    if isinstance(d, Dict) or isinstance(d, OrderedDict):
        for item in d.keys():
            if isinstance(d[item], (List, Tuple, Dict)):
                keys += 1
                k, v = count_k_v(d[item])
                values += v
                keys += k
            else:
                keys += 1
                values += 1

    elif isinstance(d, (List, Tuple)):
        for item in d:
            if isinstance(item, (List, Tuple, Dict)):
                k, v = count_k_v(item)
                values += v
                keys += k
            else:
                values += 1

    return keys, values
