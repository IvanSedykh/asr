import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    # TODO: your code here

    # base dataset returns this
    #     return {
    #     "audio": audio_wave,
    #     "spectrogram": audio_spec,
    #     "duration": audio_wave.size(1) / self.config_parser["preprocessing"]["sr"],
    #     "text": data_dict["text"],
    #     "text_encoded": self.text_encoder.encode(data_dict["text"]),
    #     "audio_path": audio_path,
    # }

    # SPECTROGRAMS
    spectrograms = [torch.squeeze(item['spectrogram'], dim=0).T for item in dataset_items]
    spectrograms_lens = [item['spectrogram'].shape[2] for item in dataset_items]
    spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
    spectrograms = spectrograms.permute(0, 2, 1)
    result_batch['spectrogram'] = spectrograms
    result_batch['spectrogram_length'] = torch.tensor(spectrograms_lens)

    # TEXTS ENCCODING
    # actually this is text char tokens
    texts_encoded = [torch.squeeze(item['text_encoded'], dim=0) for item in dataset_items]
    texts_encoded_lens = [item['text_encoded'].shape[1] for item in dataset_items]
    texts_encoded = torch.nn.utils.rnn.pad_sequence(texts_encoded, batch_first=True)
    result_batch['text_encoded'] = texts_encoded
    result_batch['text_encoded_length'] = torch.tensor(texts_encoded_lens)

    # TEXTS
    texts = [item['text'] for item in dataset_items]
    result_batch['text'] = texts

    # AUDIO PATHS
    audio_paths = [item['audio_path'] for item in dataset_items]
    result_batch['audio_path'] = audio_paths

    audios = [item['audio'] for item in dataset_items]
    result_batch['audio'] = audios

    durations = [item['duration'] for item in dataset_items]
    result_batch['duration'] = durations

    # TODO: may add other keys

    return result_batch