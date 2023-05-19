import mqtt, { MqttClient } from "mqtt";

/**
 * Class to handle the MQTT connection
 */
export class MqttHanlder {
  mqttClient: MqttClient;
  host: string;
  topic: string;
  // username: string;
  // password: string;

  /**
   * Constructor
   * @param {string} topic - The topic to subscribe to
   */
  constructor(topic: string) {
    // Use given host and port or use default
    const envHost =
      process.env.MQTT_BROKER_ADDRESS !== null &&
      process.env.MQTT_BROKER_PORT !== null;
    this.host = envHost
      ? `mqtt://${process.env.MQTT_BROKER_ADDRESS}:${process.env.MQTT_BROKER_PORT}`
      : "mqtt://localhost:1883";

    // The topic to subscribe to
    this.topic = topic;
    // this.username = process.env.MQTT_USERNAME || "";
    // this.password = process.env.MQTT_PASSWORD || "";
  }

  /**
   * Connect to the MQTT broker and subscribe to the topic
   */
  connect = () => {
    this.mqttClient = mqtt.connect(this.host);

    this.mqttClient.on("error", (error) => {
      console.log("Alert MQTT error: ", error);
    });

    this.mqttClient.on("connect", () => {
      console.log(`Alert MQTT client connected`);
      console.log(`${this.host}`);
    });

    this.mqttClient.subscribe(this.topic, { qos: 0 });

    this.mqttClient.on("message", (topic, message) => {
      console.log("Messageeee");
      console.log("Alert MQTT message: ", topic, message.toString());
    });

    this.mqttClient.on("close", () => {
      console.log(`Alert MQTT client disconnected`);
    });
  };
}
