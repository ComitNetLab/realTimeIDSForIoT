import express, { Express, Request, Response } from "express";
import { MqttHanlder } from "./mqtt/mqtt-handler";
import dotenv from "dotenv";
import statisticsRouter from "./statistics/statistics-handler";

/**
 * --------------------------------
 * Set up the server
 * --------------------------------
 */

dotenv.config();
const app: Express = express();
const port = process.env.PORT || 3000;
app.use(express.json());

/**
 * --------------------------------
 * MQTT
 * --------------------------------
 */
// Connect to the MQTT broker and subscribe to the topic
// for the alert messages
const atkDetected = new MqttHanlder("atkDetected");
atkDetected.connect();

// Connect to the MQTT broker and subscribe to the topic
// for statistics
const statistics = new MqttHanlder("atkDetection");
statistics.connect();

/**
 * --------------------------------
 * Routers
 * --------------------------------
 */
app.use("/", statisticsRouter);

// Start the server
const server = app.listen(port, () => {
  console.log(`Listening at http://localhost:${port}/`);
});
