import time
from dora import Node

node = Node()
print("Press ENTER to send a trigger...", flush=True)

try:
    while True:
        try:
            input("Press ENTER to record...")
            node.send_output("tick", b"")
            print("Sent trigger!", flush=True)
        except EOFError:
            break
except KeyboardInterrupt:
    pass
