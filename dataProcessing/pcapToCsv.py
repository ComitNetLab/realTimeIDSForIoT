import os
import csv

outputName = 'test'
fileType = '.csv'
outputPath = outputName + fileType

fileName = 'test.pcap'
os.system("tshark -r " + fileName + " -T fields "
          "-e ip.flags -e tcp.flags"  # Needs processing
          "-e ip.proto "
          "-e ip.src -e ip.dst "
          "-e tcp.srcport -e tcp.dstport "  # Needs processing
          "-e udp.srcport -e tcp.dstport "  # Needs processing
          "-e frame.len "              
          "-E separator=, -E quote=d > " + outputPath)  # add "-E header=y" to see the output headers

with open(outputPath, mode='r') as f:
    reader = csv.reader(f)


'''
LPR - Little processing required.
G   - GENERATED
---------------
Flgs	Flags de estado de flujo	Flow state flags seen in transactions                                       X LPR
flgs_number	representación numérica de los flags	Numerical representation of feature flags                   X G    
Proto	Representación de protocolo	Textual representation of transaction protocols present in network flow     X
proto_number	Representación numérica de protocolo	Numerical representation of feature proto               X G
Saddr	IP de origen	Source IP address                                                                       X
Sport	Puerto de origen	Source port number                                                                  X LPR
Daddr	IP de destino	Destination IP address                                                                  X
Dport	Puerto de destino	Destination port number                                                             X LPR
Pkts	Número total de paquetes	Total count of packets in transaction                                       
Bytes	Número total de bytes	Totan number of bytes in transaction                                            X
State	Estado de conexión	Transaction state
state_number	Representación del estado de conexión	Numerical representation of feature state               GENERATED
'''
