import struct
from struct import *
import sys


def ethernetData(socket, raw):
    # 12 bytes for source and dest mac, 2 for EtherType, then data
    dest, src, prototype = struct.unpack('! 6s 6s H', raw[:14])
    proto = socket.htons(prototype)
    data = raw[14:]
    return dest, src, proto, data
