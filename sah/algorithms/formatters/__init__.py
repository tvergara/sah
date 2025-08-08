from .gpt4all_formatter import Gpt4allFormatter
from .lima_formatter import LimaFormatter
from .xsum_formatter import XsumFormatter


def get_dataset_formatter(dataset_path: str):
    if dataset_path == 'EdinburghNLP/xsum':
        return XsumFormatter()
    if dataset_path == 'nomic-ai/gpt4all-j-prompt-generations':
        return Gpt4allFormatter()
    if dataset_path == 'GAIR/lima':
        return LimaFormatter()

    return DefaultFormatter()


class DefaultFormatter:
    def __init__(self):
        pass
    def __call__(self, example):
        return
