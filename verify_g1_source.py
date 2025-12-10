import sys
import os
import time
import socket
import struct
import threading
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from stt.audio_source import G1AudioSource

MCAST_GRP = "239.168.123.161"
MCAST_PORT = 5555


def sender():
    # Send a dummy packet
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)

    # Wait a bit for receiver to be ready
    time.sleep(1)

    # Create a dummy payload (4096 bytes of int16)
    data = np.zeros(2048, dtype=np.int16)
    # Use something recognizable? Just zeros is fine for size check.
    payload = data.tobytes()

    print(f"Sender: Sending {len(payload)} bytes to {MCAST_GRP}:{MCAST_PORT}")
    sock.sendto(payload, (MCAST_GRP, MCAST_PORT))

    time.sleep(0.5)
    print(f"Sender: Sending {len(payload)} bytes to 127.0.0.1:{MCAST_PORT}")
    sock.sendto(payload, ("127.0.0.1", MCAST_PORT))

    sock.close()


def test_g1_source():
    print("Testing G1AudioSource...")

    # Start sender thread
    t = threading.Thread(target=sender)
    t.start()

    try:
        with G1AudioSource() as source:
            print("G1AudioSource entered context.")
            print("Reading...")
            # Set a timeout on the socket to avoid hanging forever if multicast fails
            if source.sock:
                source.sock.settimeout(3.0)

            data = source.read()
            print(f"Read {len(data)} bytes.")

            if len(data) == 4096:
                print("SUCCESS: Read correct amount of data.")
            else:
                print("FAILURE: Did not read expected amount of data or timed out.")

    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        t.join()


if __name__ == "__main__":
    test_g1_source()
