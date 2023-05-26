import sys
import paho.mqtt.client as mqtt
import time
from itertools import zip_longest

DEBUG=False

# ------------------ argv validation ------------------
if(len(sys.argv) < 2):
    print('Please provide the broker address as a command-line argument.')
    sys.exit(1)
if(len(sys.argv) < 4):
    print('Please provide the path to the 2 CSVs files as a command-line argument.')


# ------------------ MQTT ------------------
brokerAddress=sys.argv[1]
brokerPort=1883
username='<username>'
password='<password>'

def on_connect(client, userdata, flags, rc):
    if(DEBUG):
        print('Connected with result code ' + str(rc))

def post_data_frame(data):
    client.publish('atkDetection', data)

client = mqtt.Client()

# set connection
# client.username_pw_set(username, password)
client.connect(brokerAddress, brokerPort)

client.on_connect = on_connect

client.loop_start()

# ------------------ CSV ------------------
filePath=sys.argv[2]
file2Path=sys.argv[3]

'''
Read the files of the attacks of the models and send them to the broker.
'''
with open(filePath, 'r') as file1, open(file2Path, 'r') as file2:
    for line1, line2 in zip_longest(file1, file2, fillvalue=None):
        if line1 is not None:
            post_data_frame(line1)
        if line2 is not None:
            post_data_frame(line2)
        time.sleep(0.2) # 200 ms

client.disconnect()
print('Mock done')