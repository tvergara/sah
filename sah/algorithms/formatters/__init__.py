from .gpt4all_formatter import Gpt4allFormatter
from .lima_formatter import LimaFormatter
from .meta_math_formatter import MetaMathFormatter
from .open_thoughts_formatter import OpenThoughtsFormatter
from .xsum_formatter import XsumFormatter


def get_dataset_formatter(dataset_path: str):
    if dataset_path == 'EdinburghNLP/xsum':
        return XsumFormatter()
    if dataset_path == 'nomic-ai/gpt4all-j-prompt-generations':
        return Gpt4allFormatter()
    if dataset_path == 'GAIR/lima':
        return LimaFormatter()
    if dataset_path == "meta-math/MetaMathQA":
        return MetaMathFormatter()
    if dataset_path == "open-thoughts/OpenThoughts3-1.2M":
        return OpenThoughtsFormatter()

    return DefaultFormatter()


class DefaultFormatter:
    def __init__(self):
        pass
    def __call__(self, example):
        return
