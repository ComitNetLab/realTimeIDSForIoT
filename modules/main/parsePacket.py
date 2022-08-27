import struct
import socket
import binascii


def ethernetData(raw):
    # 12 bytes for source and dest mac, 2 for EtherType, then data
    dest, src, prototype = struct.unpack('! 6s 6s H', raw[:14])
    dest = getMacAddr(dest)
    src = getMacAddr(src)
    proto = socket.htons(prototype)
    data = raw[14:]
    return dest, src, proto, data


def ipv4Head(raw):
    version_header_l = raw[0]
    version = version_header_l >> 4
    header_l = (version_header_l & 15) * 4
    ttl, proto, src, target = struct.unpack('! 8x B B 2x 4s 4s', raw[:20])
    src = getIp(src)
    target = getIp(target)
    data = raw[header_l:]
    return version, header_l, ttl, proto, src, target, data


def getIp(addr):
    return '.'.join(map(str, addr))


def getMacAddr(addr):
    addr = str(binascii.hexlify(addr))[2:-1]
    print(addr)
    print(type(addr))
    return ':'.join(addr[i:i+2] for i in range(0, len(addr), 2))
