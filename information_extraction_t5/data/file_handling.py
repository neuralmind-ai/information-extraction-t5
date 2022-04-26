"""Tools for handling dataset files."""
import glob
import json
import tarfile
from typing import Tuple


def decompress(fname):
    """Unpack a tar file and return the name of the JSON dataset file.

    Args:
        fname: compressed dataset file name

    Returns:
        The name of the unpacked JSON raw dataset file.
    """
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall('data/raw/')
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        tar.extractall('data/raw/')
        tar.close()

    fname = glob.glob('data/raw/*json')[-1]

    return fname


def load_raw_data(fname: str) -> Tuple[dict, list, str]:
    """Loads raw dataset file.

    Args:
        fname: the dataset file name

    Returns:
        A tuple with the json-like raw data dict and a corresponding list
            of tuples with keys and values.
    """
    if fname.endswith('tar') or fname.endswith('tar.gz'):
        print(f'>> Decompressing dataset file {fname}...')
        raw_data_fname = decompress(fname)
    else:
        raw_data_fname = fname

    with open(raw_data_fname) as f:
        raw_data = json.load(f)
        documents = list(raw_data.items())

    return raw_data, documents, raw_data_fname
