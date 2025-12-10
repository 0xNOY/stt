import abc
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from loguru import logger
from faster_whisper import WhisperModel

from stt.audio_source import AbstractAudioSource


@dataclass
class TranscriberConfig:
    model_name: str
    device: str
    compute_type: str
    batch_size: int
    language: str | None


class Transcriber(abc.ABC):
    def __init__(self, config: TranscriberConfig):
        self.config = config

    @abc.abstractmethod
    def transcribe(
        self, audio: np.ndarray[np.int16], language: str | None = None
    ) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def transcribe_stream(self, audio_source: AbstractAudioSource) -> Iterator[str]:
        raise NotImplementedError


class WhisperTranscriber(Transcriber):
    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self.model = WhisperModel(
            config.model_name, device=config.device, compute_type=config.compute_type
        )

    def transcribe(
        self, audio: np.ndarray[np.int16], language: str | None = None
    ) -> str:
        segs, info = self.model.transcribe(
            audio,
            batch_size=self.config.batch_size,
            language=language or self.config.language,
        )

        logger.debug(f"Transcribed({info.duration:.2f}s): {segs[0].text}")

        return segs[0].text
