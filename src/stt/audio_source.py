import abc
import socket
import struct
import typing

import numpy as np
from loguru import logger

# Optional dependency check for netifaces
try:
    import netifaces
except ImportError:
    netifaces = None


class AbstractAudioSource(abc.ABC):
    """
    Abstract base class for audio sources.
    Defines the interface for reading audio data from various sources (e.g., microphone, network stream).
    """

    def __init__(self, sample_rate: int, channels: int, chunk_size: int):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def __enter__(self) -> "AbstractAudioSource":
        """Open the stream and initialize resources"""
        raise NotImplementedError

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the stream and release resources"""
        raise NotImplementedError

    @abc.abstractmethod
    def read(self) -> np.ndarray[np.int16]:
        """Read an audio chunk and return it as bytes

        Returns:
            np.ndarray[np.int16]: Read 16-bit little-endian audio data
        """
        raise NotImplementedError


class G1AudioSource(AbstractAudioSource):
    """
    Unitree G1 robot microphone audio receiver class.
    Receives audio data using multicast UDP.
    """

    DEFAULT_MULTICAST_IP = "239.168.123.161"
    DEFAULT_PORT = 5555
    DEFAULT_SAMPLE_RATE = 16000
    DEFAULT_CHANNELS = 1
    DEFAULT_CHUNK_SIZE = 4096

    def __init__(
        self,
        multicast_ip: str = DEFAULT_MULTICAST_IP,
        port: int = DEFAULT_PORT,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        super().__init__(
            sample_rate=self.DEFAULT_SAMPLE_RATE,
            channels=self.DEFAULT_CHANNELS,
            chunk_size=chunk_size,
        )
        self.multicast_ip = multicast_ip
        self.port = port
        self.sock: typing.Optional[socket.socket] = None

    def _get_local_ip(self, target_subnet_prefix="") -> str:
        """Finds the local IP address on the specific subnet."""
        if not target_subnet_prefix:
            target_subnet_prefix = self.multicast_ip[: self.multicast_ip.rfind(".") + 1]

        if netifaces:
            try:
                for interface in netifaces.interfaces():
                    addrs = netifaces.ifaddresses(interface)
                    if netifaces.AF_INET in addrs:
                        for addr_info in addrs[netifaces.AF_INET]:
                            ip = addr_info["addr"]
                            if ip.startswith(target_subnet_prefix):
                                return ip
            except Exception:
                logger.warning("Failed to get local IP address.")
                pass
        else:
            logger.warning("netifaces not installed.")

        return "0.0.0.0"

    def __enter__(self) -> "G1AudioSource":
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)

        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except AttributeError:
            pass

        self.sock.bind(("", self.port))

        local_ip = self._get_local_ip()

        try:
            if local_ip != "0.0.0.0":
                mreq = struct.pack(
                    "4s4s",
                    socket.inet_aton(self.multicast_ip),
                    socket.inet_aton(local_ip),
                )
                self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            else:
                mreq = struct.pack(
                    "4s4s",
                    socket.inet_aton(self.multicast_ip),
                    socket.inet_aton("0.0.0.0"),
                )
                self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        except Exception as e:
            if self.sock:
                self.sock.close()
                self.sock = None
            raise RuntimeError(f"Failed to join multicast group: {e}") from e

        logger.info(f"Listening for audio on {self.multicast_ip}:{self.port}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sock:
            self.sock.close()
            self.sock = None

    def read(self) -> np.ndarray[np.int16]:
        if self.sock is None:
            raise RuntimeError("Audio source is not open. Use 'with' statement.")

        try:
            data, _ = self.sock.recvfrom(self.chunk_size)
            return np.frombuffer(data, dtype=np.int16)
        except socket.timeout:
            logger.debug("Timeout while receiving data")
            return np.array([], dtype=np.int16)
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            return np.array([], dtype=np.int16)
