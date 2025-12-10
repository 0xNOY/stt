import typing
from loguru import logger
import httpx


def request_stt(
    host: str = "127.0.0.1",
    port: int = 8080,
    max_duration_sec: typing.Optional[float] = None,
    silence_threshold_sec: typing.Optional[float] = None,
    speech_confidence_threshold: typing.Optional[float] = None,
    pre_buffer_sec: typing.Optional[float] = None,
    language: typing.Optional[str] = None,
    timeout: typing.Optional[float] = None,
) -> str:
    """
    Transcribe audio by making a request to the STT HTTP node.

    Args:
        host: The host of the STT HTTP node.
        port: The port of the STT HTTP node.
        max_duration_sec: Maximum duration of recording in seconds.
        silence_threshold_sec: Silence threshold in seconds to stop recording.
        speech_confidence_threshold: Confidence threshold for speech detection (0.0 - 1.0).
        pre_buffer_sec: Pre-buffer duration in seconds.
        language: Language code (e.g. "en", "ja") or None to auto-detect.
        timeout: Request timeout in seconds. Defaults to None (indefinite) as recording can be long.

    Returns:
        The transcribed text.

    Raises:
        httpx.HTTPStatusError: If the server returns an error status code.
        httpx.RequestError: If an error occurs while requesting.
    """
    url = f"http://{host}:{port}/transcribe"

    payload: typing.Dict[str, typing.Any] = {}

    recorder_config = {}
    if max_duration_sec is not None:
        recorder_config["max_duration_sec"] = max_duration_sec
    if silence_threshold_sec is not None:
        recorder_config["silence_threshold_sec"] = silence_threshold_sec
    if speech_confidence_threshold is not None:
        recorder_config["speech_confidence_threshold"] = speech_confidence_threshold
    if pre_buffer_sec is not None:
        recorder_config["pre_buffer_sec"] = pre_buffer_sec

    if recorder_config:
        payload["recorder"] = recorder_config

    transcriber_runtime = {}
    if language is not None:
        transcriber_runtime["language"] = language

    if transcriber_runtime:
        payload["transcriber"] = transcriber_runtime

    logger.debug(f"Requesting transcription from {url} with payload: {payload}")

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("text", "")
