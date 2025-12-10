import time
from collections import deque

import torch
import numpy as np
from loguru import logger
from silero_vad import load_silero_vad

from stt.audio_source import AbstractAudioSource


MAX_INT16 = np.iinfo(np.int16).max


class VoiceActivityRecorder:
    """
    A class for recording audio based on voice activity detection (VAD).

    This recorder listens to an audio source and captures audio segments where speech
    is detected using the Silero VAD model.
    """

    def __init__(self, audio_source: AbstractAudioSource):
        self.audio_source = audio_source
        self.vad_model = load_silero_vad()
        self.sample_rate = audio_source.sample_rate
        self.chunk_size = audio_source.chunk_size

    def record_on_voice_activity(
        self,
        max_duration_sec: float,
        silence_threshold_sec: float,
        speech_confidence_threshold: float,
        pre_buffer_sec: float = 0.2,
    ) -> np.ndarray[np.int16] | None:
        """
        Records audio segments where speech is detected using the Silero VAD model.

        Args:
            max_duration_sec: Maximum recording duration in seconds.
            silence_threshold_sec: Number of seconds of silence to consider the recording finished.
            speech_confidence_threshold: VAD confidence threshold. Values above this are considered speech.
            pre_buffer_sec: Number of seconds of audio to include before the start of speech.

        Returns:
            NumPy array containing the recorded audio data. Returns None if no recording was made.
        """
        chunks_per_sec = self.sample_rate / self.chunk_size
        pre_buffer_size = int(pre_buffer_sec * chunks_per_sec)
        silent_chunk_threshold = int(silence_threshold_sec * chunks_per_sec)

        pre_buffer = deque(maxlen=pre_buffer_size)
        recorded_chunks: list[np.ndarray] = []

        is_recording = False
        silent_chunk_count = 0
        recording_start_time = time.time()

        try:
            with self.audio_source as source:
                while time.time() - recording_start_time < max_duration_sec:
                    chunk_data = source.read()
                    if not chunk_data.size:
                        logger.debug("Empty chunk received.")
                        continue

                    audio_chunk_tensor = (
                        torch.from_numpy(chunk_data).float() / MAX_INT16
                    )

                    if not is_recording:
                        pre_buffer.append(chunk_data)

                    speech_probability = self.vad_model(
                        audio_chunk_tensor, self.sample_rate
                    ).item()

                    is_speaking = speech_probability > speech_confidence_threshold

                    if is_speaking and not is_recording:
                        logger.info("Voice detected, starting to record.")
                        is_recording = True
                        recorded_chunks.extend(list(pre_buffer))

                    if is_recording:
                        recorded_chunks.append(chunk_data)
                        if not is_speaking:
                            silent_chunk_count += 1
                            if silent_chunk_count > silent_chunk_threshold:
                                logger.info("Silence detected, stopping recording.")
                                break
                        else:
                            silent_chunk_count = 0

        except KeyboardInterrupt:
            logger.info("Recording stopped by user.")
        except Exception as e:
            logger.error(f"An error occurred during recording: {e}")
        finally:
            logger.info("Recording finished.")

        if not recorded_chunks:
            return None

        return np.concatenate(recorded_chunks)
