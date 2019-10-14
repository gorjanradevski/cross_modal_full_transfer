import torch
from typing import Tuple


def collate_pad_batch(batch: Tuple[torch.Tensor, torch.Tensor]):
    images, sentences = zip(*batch)
    images = torch.stack(images, 0)
    padded_sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True)

    return images, padded_sentences
