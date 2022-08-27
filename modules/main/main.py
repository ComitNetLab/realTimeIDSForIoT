import socket
import platform
from parsePacket import *

# Initial config
system = platform.system()
if system.startswith('Win'):
    packetInterface = socket.AF_INET
elif system.startswith('Lin'):
    packetInterface = socket.PF_PACKET
else:
    print('Unkown system, exit script')
    sys.exit()

s = socket.socket(packetInterface, socket.SOCK_RAW, socket.IPPROTO_RAW)
while True:
    raw, add = s.recvfrom(65535)
    print(ethernetData(s, raw))


