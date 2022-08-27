import platform
from parsePacket import *
import sys

# delete later
import time

print("Init program")
# Initial config
system = platform.system()
if system.startswith('Win'):
    print("Windows Recognized")
    packetInterface = socket.AF_INET
elif system.startswith('Lin'):
    print("Linux recognized")
    packetInterface = socket.PF_PACKET
else:
    print('Unkown system, exit script')
    sys.exit()

print("Creating socket")
s = socket.socket(packetInterface, socket.SOCK_RAW, socket.ntohs(3))
print("Start loop")
while True:
    raw, add = s.recvfrom(65535)
    eth = ethernetData(raw)
    print('Destination: {}, Source: {}, Protocol: {}'.format(eth[0], eth[1], eth[2]))
    if eth[2] == 8:  # ipv4
        ipv4 = ipv4Head(eth[3])
        print('\t - ' + 'IPv4 Packet:')
        print('\t\t - ' + 'Version: {}, Header Length: {}, TTL: {}, '.format(ipv4[0], ipv4[1], ipv4[2]))
        print('\t\t - ' + 'Protocol: {}, Source: {}, Target: {}'.format(ipv4[3], ipv4[4], ipv4[5]))
        '''
        TCP 6
        ICMP 1
        UDP 17
        RDP 27
        '''
    time.sleep(5)

