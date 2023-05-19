import sys
import joblib
import paho.mqtt.client as mqtt
from realTimeIDSForIoT.modules.detection.ValidModels import AttackTypes, get_model_function

# ------------------ argv validation ------------------
if len(sys.argv) < 2:
    print("Please provide the name of the joblib AI model saved in ./models dir as a command-line argument.")
    sys.exit(1)
if len(sys.argv) < 3:
    print("Please provide the broker address as a command-line argument.")
    sys.exit(1)


# ------------------ Model ------------------
# Get the model name from the command-line argument
modelName=sys.argv[1]

# Load the model
model=joblib.load('./models/' + modelName)


# ------------------ MQTT ------------------
brokerAddress=sys.argv[2]
brokerPort=1883

username='<username>'
password='<password>'

'''
Callback function for MQTT connection event
subscribes the module to the topic 'atkDetection' 
'''
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("atkDetection")

'''
Callback function for MQTT message event
Retrieves the payload from the message and calls the processing function
depending on the model name
'''
def on_message(client, userdata, msg):
    payload = msg.payload.decode('utf-8')
    processing_function = get_model_function(modelName.split('.')[0])
    processing_function(payload, model)

'''
Publishes the attack type to the topic 'intrusionDetected'
'''
def alert_intrusion(atk_type: AttackTypes, recorded_data: str):
    alert_message = atk_type.value + ':' + recorded_data
    client.publish('intrusionDetected', alert_message)


# ------------------ Main ------------------
client = mqtt.Client()

# set connection
# client.username_pw_set(username, password)
client.connect(brokerAddress, brokerPort)

# set callback functions
client.on_connect = on_connect
client.on_message = on_message

# Start MQTT network loop and wait for incoming messages
client.loop_start()

# Keep the program running
while True:
    pass