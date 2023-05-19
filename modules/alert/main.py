import sys
import paho.mqtt.client as mqtt

if len(sys.argv) < 2:
    print("Please provide the broker address as a command-line argument.")
    sys.exit(1)

# ------------------ MQTT ------------------
brokerAddress=sys.argv[1]
brokerPort=1883

username='<username>'
password='<password>'

'''
Callback function for MQTT connection event
subscribes the module to the topic 'atkDetected' 
'''
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("atkDetected")

'''
Callback function for MQTT message event
Retrieves the payload from the message and starts
the processing function depending on the attack type
'''
def on_message(client, userdata, msg):
    payload = msg.payload.decode('utf-8')
    atk_type, data = payload.split(':')

    print(payload)

# ------------------ Main ------------------
client = mqtt.Client()
client.connect(brokerAddress, brokerPort)

# set callback functions
client.on_connect = on_connect
client.on_message = on_message

client.loop_start()

# keep the program running
while True:
    pass
