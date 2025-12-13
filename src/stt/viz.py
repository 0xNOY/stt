import threading
import time
import typing
from functools import partial

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.server.server import Server
from loguru import logger

from stt.audio_source import AbstractAudioSource, G1AudioSource


class MockAudioSource(AbstractAudioSource):
    def __init__(self, chunk_size=4096, sample_rate=16000):
        super().__init__(sample_rate, 1, chunk_size)
        self.t = 0
        self.running = True

    def __enter__(self):
        self.running = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False

    def read(self) -> np.ndarray:
        # Simulate ~16ms of audio (CHUNK_SIZE/SAMPLE_RATE approx 0.25s? No wait)
        # 4096 / 16000 = 0.256s.
        time.sleep(self.chunk_size / self.sample_rate)
        if not self.running:
            return np.array([], dtype=np.int16)

        # Generate sine wave
        t = np.arange(self.chunk_size) + self.t
        self.t += self.chunk_size
        freq = 440.0
        audio = 32768.0 * 0.5 * np.sin(2 * np.pi * freq * t / self.sample_rate)
        # Add some noise
        noise = np.random.normal(0, 1000, self.chunk_size)
        return (audio + noise).astype(np.int16)


# Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 4096
WAVEFORM_WINDOW_SIZE = SAMPLE_RATE * 5  # Show last 5 seconds
HISTORY_SIZE = 1000  # Number of packets to show stats for


class AudioMonitor:
    def __init__(self):
        self.source_wave = ColumnDataSource(data=dict(x=[], y=[]))
        self.source_stats = ColumnDataSource(data=dict(time=[], interval=[], size=[]))

        self.audio_buffer = np.zeros(WAVEFORM_WINDOW_SIZE, dtype=np.float32)
        self.last_packet_time = 0
        self.packet_count = 0

        self.running = False
        self.thread: typing.Optional[threading.Thread] = None

    def update_waveform(self, new_data: np.ndarray):
        # Roll buffer and add new data
        self.audio_buffer = np.roll(self.audio_buffer, -len(new_data))
        self.audio_buffer[-len(new_data) :] = new_data

        # Downsample for visualization if needed/performance heavy
        # For 5s @ 16kHz = 80k points. Bokeh handles this okay usually, but decimation is safer.
        # Let's plot raw for now, or maybe stride=10
        stride = 10
        x = np.arange(0, len(self.audio_buffer), stride) / SAMPLE_RATE
        y = self.audio_buffer[::stride]

        self.source_wave.data = dict(x=x, y=y)

    def update_stats(self, receive_time, interval, size):
        new_data = dict(
            time=[receive_time], interval=[interval * 1000], size=[size]
        )  # Interval in ms
        self.source_stats.stream(new_data, rollover=HISTORY_SIZE)

    def start_capture_thread(self, doc, audio_source_cls):
        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop, args=(doc, audio_source_cls), daemon=True
        )
        self.thread.start()

    def _capture_loop(self, doc, audio_source_cls):
        logger.info(f"Starting audio capture thread with {audio_source_cls.__name__}")
        with audio_source_cls(chunk_size=CHUNK_SIZE) as audio_source:
            self.last_packet_time = time.time()

            while self.running:
                try:
                    audio_data = audio_source.read()
                    current_time = time.time()

                    if len(audio_data) > 0:
                        # Normalize to -1..1
                        normalized_data = audio_data.astype(np.float32) / 32768.0

                        interval = current_time - self.last_packet_time
                        self.last_packet_time = current_time

                        doc.add_next_tick_callback(
                            partial(self.update_waveform, normalized_data)
                        )
                        doc.add_next_tick_callback(
                            partial(
                                self.update_stats,
                                self.packet_count,
                                interval,
                                len(audio_data),
                            )
                        )
                        self.packet_count += 1
                    else:
                        # Handle potential timeouts or empty reads if needed
                        pass

                except Exception as e:
                    logger.error(f"Error in capture loop: {e}")
                    time.sleep(0.1)


def make_document(doc, source_cls=G1AudioSource):
    monitor = AudioMonitor()

    # Waveform Plot
    p_wave = figure(
        title="Real-time Audio Waveform (Last 5s)",
        height=300,
        sizing_mode="stretch_width",
        y_range=(-1.1, 1.1),
    )
    p_wave.line("x", "y", source=monitor.source_wave, line_width=1)
    p_wave.xaxis.axis_label = "Time (s)"
    p_wave.yaxis.axis_label = "Amplitude"

    # Inter-arrival Time Plot
    p_interval = figure(
        title="Packet Inter-arrival Time (ms)", height=200, sizing_mode="stretch_width"
    )
    p_interval.scatter(
        "time", "interval", source=monitor.source_stats, size=5, color="navy", alpha=0.5
    )
    p_interval.line(
        "time", "interval", source=monitor.source_stats, color="navy", alpha=0.3
    )
    p_interval.xaxis.axis_label = "Packet Count"
    p_interval.yaxis.axis_label = "Interval (ms)"

    # Packet Size Plot
    p_size = figure(
        title="Packet Size (bytes)", height=200, sizing_mode="stretch_width"
    )
    p_size.scatter(
        "time",
        "size",
        source=monitor.source_stats,
        size=5,
        color="firebrick",
        alpha=0.5,
    )
    p_size.xaxis.axis_label = "Packet Count"
    p_size.yaxis.axis_label = "Size (bytes)"

    layout = column(p_wave, p_interval, p_size, sizing_mode="stretch_width")
    doc.add_root(layout)
    doc.title = "G1 Audio Monitor"

    monitor.start_capture_thread(doc, source_cls)


def start(mock: bool = False):
    source_cls = MockAudioSource if mock else G1AudioSource

    server = Server({"/": partial(make_document, source_cls=source_cls)}, num_procs=1)

    server.start()

    logger.info(f"Opening Bokeh application on http://localhost:{server.port}/")
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()


if __name__ == "__main__":
    start()
