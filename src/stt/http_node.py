import contextlib
import os
import typing

import torch
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel, Field

from stt.audio_source import G1AudioSource
from stt.recoder import VoiceActivityRecorder
from stt.transcriber import TranscriberConfig, WhisperTranscriber


class AudioSourceConfig(BaseModel):
    multicast_ip: str = Field(
        default="239.168.123.161", description="Multicast IP for audio source"
    )
    port: int = Field(default=5555, description="Port for audio source")
    chunk_size: int = Field(default=4096, description="Chunk size for audio reading")


class RecorderRuntimeConfig(BaseModel):
    max_duration_sec: float = Field(
        default=15.0, description="Maximum recording duration"
    )
    silence_threshold_sec: float = Field(
        default=1.5, description="Silence duration to stop recording"
    )
    speech_confidence_threshold: float = Field(default=0.4, description="VAD threshold")
    pre_buffer_sec: float = Field(default=0.1, description="Pre-buffer duration")


class TranscriberInitConfig(BaseModel):
    model_name: str = Field(default="distil-large-v3", description="Whisper model name")
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run model on",
    )
    compute_type: str = Field(
        default="int8_float16" if torch.cuda.is_available() else "int8",
        description="Compute type",
    )
    batch_size: int = Field(default=8, description="Batch size")


class TranscriberRuntimeConfig(BaseModel):
    language: typing.Optional[str] = Field(default=None, description="Language code")


class GlobalConfig(BaseModel):
    source: AudioSourceConfig = Field(default_factory=AudioSourceConfig)
    recorder: RecorderRuntimeConfig = Field(default_factory=RecorderRuntimeConfig)
    transcriber_init: TranscriberInitConfig = Field(
        default_factory=TranscriberInitConfig
    )
    transcriber_runtime: TranscriberRuntimeConfig = Field(
        default_factory=TranscriberRuntimeConfig
    )


class TranscriptionRequest(BaseModel):
    recorder: typing.Optional[RecorderRuntimeConfig] = None
    transcriber: typing.Optional[TranscriberRuntimeConfig] = None


class TranscriptionResponse(BaseModel):
    text: str


class AppState:
    def __init__(self):
        self.config = GlobalConfig()
        self.source: typing.Optional[G1AudioSource] = None
        self.recorder: typing.Optional[VoiceActivityRecorder] = None
        self.transcriber: typing.Optional[WhisperTranscriber] = None


state = AppState()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize components on startup
    logger.info("Initializing STT Node...")

    # 1. Audio Source
    state.source = G1AudioSource(
        multicast_ip=state.config.source.multicast_ip,
        port=state.config.source.port,
        chunk_size=state.config.source.chunk_size,
    )

    # 2. Recorder
    # Note: Recorder takes source in __init__, but we'll pass the instance.
    state.recorder = VoiceActivityRecorder(state.source)

    # 3. Transcriber
    transcriber_config = TranscriberConfig(
        model_name=state.config.transcriber_init.model_name,
        device=state.config.transcriber_init.device,
        compute_type=state.config.transcriber_init.compute_type,
        batch_size=state.config.transcriber_init.batch_size,
        language=state.config.transcriber_runtime.language,  # Default language
    )
    state.transcriber = WhisperTranscriber(transcriber_config)

    logger.info("STT Node Initialized.")
    yield
    # Cleanup
    logger.info("Shutting down STT Node...")
    # Add any cleanup if necessary


app = FastAPI(lifespan=lifespan)


@app.post("/transcribe", response_model=TranscriptionResponse)
def transcribe_audio(request: TranscriptionRequest):
    logger.info(f"Received transcription request: {request}")

    # Merge defaults with request overrides
    recorder_config = request.recorder or state.config.recorder
    transcriber_runtime = request.transcriber or state.config.transcriber_runtime

    if not state.recorder or not state.transcriber:
        raise RuntimeError("Components not initialized properly.")

    # Record
    logger.info("Starting recording...")
    audio_data = state.recorder.record_on_voice_activity(
        max_duration_sec=recorder_config.max_duration_sec,
        silence_threshold_sec=recorder_config.silence_threshold_sec,
        speech_confidence_threshold=recorder_config.speech_confidence_threshold,
        pre_buffer_sec=recorder_config.pre_buffer_sec,
    )

    if audio_data is None:
        logger.info("No speech detected.")
        return TranscriptionResponse(text="")

    logger.info(f"Recorded {len(audio_data) / state.source.sample_rate:.2f}s audio.")

    # Transcribe
    logger.info("Transcribing...")
    text = state.transcriber.transcribe(
        audio_data, language=transcriber_runtime.language
    )

    logger.info(f"Result: {text}")
    return TranscriptionResponse(text=text)


def start():
    import uvicorn

    host = os.getenv("STT_HTTP_NODE_HOST", "0.0.0.0")
    port = os.getenv("STT_HTTP_NODE_PORT", 8080)
    uvicorn.run("stt.http_node:app", host=host, port=port)
