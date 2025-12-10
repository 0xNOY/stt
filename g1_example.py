import asyncio
import os
import socket
import struct
import sys
import numpy as np
import collections
import time
import torch
import omegaconf
import typing
import pyannote.audio.core.model
import pyannote.audio.core.task

# Fix for "WeightsUnpickler error"
# This is required for PyTorch 2.6+ when loading checkpoints containing omegaconf objects
torch.serialization.add_safe_globals(
    [
        omegaconf.listconfig.ListConfig,
        omegaconf.dictconfig.DictConfig,
        omegaconf.base.ContainerMetadata,
        typing.Any,
        list,
        dict,
        set,
        tuple,
        str,
        int,
        float,
        bool,
        type(None),
        collections.defaultdict,
        collections.OrderedDict,
        omegaconf.nodes.AnyNode,
        omegaconf.nodes.IntegerNode,
        omegaconf.nodes.StringNode,
        omegaconf.nodes.BooleanNode,
        omegaconf.nodes.FloatNode,
        omegaconf.nodes.ValueNode,
        omegaconf.nodes.InterpolationResultNode,
        omegaconf.base.Metadata,
        torch.torch_version.TorchVersion,
        pyannote.audio.core.model.Introspection,
        pyannote.audio.core.task.Specifications,
        pyannote.audio.core.task.Task,
        pyannote.audio.core.task.Problem,
        pyannote.audio.core.task.Resolution,
        # pyannote.audio.core.task.Method
    ]
)

import whisperx

import netifaces

# G1 Audio Configuration
MULTICAST_IP = "239.168.123.161"
PORT = 5555
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = np.int16

# VAD Configuration
SILERO_THRESHOLD = 0.5
SILENCE_LIMIT_SEC = 2.0  # Seconds of silence to trigger transcription
VAD_FRAME_SIZE = 512  # Silero supports 512, 1024, 1536 samples at 16000Hz

# WhisperX Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"
BATCH_SIZE = 16


def get_local_ip(target_subnet_prefix="192.168.123."):
    """Finds the local IP address on the specific subnet."""
    try:
        import netifaces

        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr_info in addrs[netifaces.AF_INET]:
                    ip = addr_info["addr"]
                    if ip.startswith(target_subnet_prefix):
                        return ip
    except ImportError:
        # Fallback if netifaces not installed, though we should probably install it.
        # But wait, python standard lib doesn't make this easy.
        # Let's try a socket join trick
        pass

    return "0.0.0.0"  # Fallback


class AudioReceiver:
    def __init__(self, multicast_ip, port):
        self.multicast_ip = multicast_ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        # Allow reusing the address
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except AttributeError:
            pass

        # Bind to the port
        self.sock.bind(("", self.port))

        # Determine interface IP
        local_ip = get_local_ip()
        if local_ip == "0.0.0.0":
            print(
                "WARNING: Could not find interface on 192.168.123.x. Multicast might fail."
            )
        else:
            print(f"DEBUG: Found local ingest interface: {local_ip}")

        # Join multicast group
        # struct ip_mreq { struct in_addr imr_multiaddr; struct in_addr imr_interface; };
        try:
            mreq = struct.pack(
                "4s4s", socket.inet_aton(self.multicast_ip), socket.inet_aton(local_ip)
            )
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except Exception as e:
            print(f"Error joining multicast group with specific interface: {e}")
            # Fallback to default behavior
            mreq = struct.pack(
                "4sl", socket.inet_aton(self.multicast_ip), socket.INADDR_ANY
            )
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        print(f"Listening for audio on {self.multicast_ip}:{self.port}")

    def recv_chunk(self):
        try:
            print("DEBUG: Receiving chunk...")
            data, _ = self.sock.recvfrom(4096)
            # data is raw bytes. Convert to numpy array
            audio_data = np.frombuffer(data, dtype=DTYPE)
            return audio_data
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None


class G1WhisperClient:
    def __init__(self):
        # Initialize Silero VAD
        print("Loading Silero VAD model...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = utils
        self.vad_model.to(DEVICE)
        print("Silero VAD model loaded.")

        # Initialize WhisperX
        print(f"Loading WhisperX model on {DEVICE}...")
        self.model = whisperx.load_model("large-v2", DEVICE, compute_type=COMPUTE_TYPE)
        print("WhisperX model loaded.")

        self.receiver = AudioReceiver(MULTICAST_IP, PORT)
        self.buffer = []
        self.voice_buffer = []  # Holds audio during voice activity
        self.is_speaking = False
        self.silence_start_time = None
        self.frame_buffer = np.array([], dtype=DTYPE)

    def process_audio(self):
        print("Starting processing loop... Speak into the G1 microphone.")

        try:
            while True:
                chunk = self.receiver.recv_chunk()
                if chunk is None:
                    print("DEBUG: No chunk received")
                    continue

                # Append chunk to a temporary frame buffer
                self.frame_buffer = np.concatenate((self.frame_buffer, chunk))
                print(f"DEBUG: Frame buffer size: {len(self.frame_buffer)}")

                # Check if we have enough data for VAD processing frame
                # Silero expects float tensor
                while len(self.frame_buffer) >= VAD_FRAME_SIZE:
                    frame = self.frame_buffer[:VAD_FRAME_SIZE]
                    self.frame_buffer = self.frame_buffer[VAD_FRAME_SIZE:]

                    # Process VAD
                    # Convert to float32 tensor normalized to [-1, 1]
                    frame_float = frame.astype(np.float32) / 32768.0
                    tensor = torch.from_numpy(frame_float).to(DEVICE)

                    # Silero model expects (batch, samples) if > 1 dim, or just samples.
                    # It returns probability of speech
                    voice_prob = self.vad_model(tensor, SAMPLE_RATE).item()

                    print(f"DEBUG: Voice probability: {voice_prob}")

                    if voice_prob >= SILERO_THRESHOLD:
                        if not self.is_speaking:
                            print("Voice detected...")
                            self.is_speaking = True
                        self.voice_buffer.append(frame)
                        self.silence_start_time = None
                    else:
                        if self.is_speaking:
                            # We were speaking, now it's silent. Check for how long.
                            self.voice_buffer.append(
                                frame
                            )  # Keep appending to catch trails

                            if self.silence_start_time is None:
                                self.silence_start_time = time.time()

                            elapsed_silence = time.time() - self.silence_start_time
                            if elapsed_silence > SILENCE_LIMIT_SEC:
                                print(
                                    f"Silence detected ({elapsed_silence:.1f}s). Transcribing..."
                                )
                                self.transcribe()
                                self.is_speaking = False
                                self.voice_buffer = []
                                self.silence_start_time = None
                                return

        except KeyboardInterrupt:
            print("\nStopping...")

    def transcribe(self):
        if not self.voice_buffer:
            return

        # Concatenate all frames
        audio_data = np.concatenate(self.voice_buffer)

        # Convert to float32 and normalize for Whisper
        audio_float = audio_data.astype(np.float32) / 32768.0

        try:
            # WhisperX expects audio as numpy array
            result = self.model.transcribe(audio_float, batch_size=BATCH_SIZE)

            # Align (Optional, skipping for speed unless requested)

            for segment in result["segments"]:
                print(f"HOST: {segment['text']}")

        except Exception as e:
            print(f"Transcription error: {e}")


if __name__ == "__main__":
    try:
        client = G1WhisperClient()
        client.process_audio()
    except Exception as e:
        print(f"Fatal error: {e}")
