import express, { Express, Request, Response } from "express";
import dotenv from "dotenv";
import { MqttHanlder } from "./mqtt/mqtt-handler.js";
import statisticsRouter from "./statistics/statistics-handler.js";
import { connectToDatabase } from "./database/DatabaseConn.js";

/**
 * --------------------------------
 * Set up the server
 * --------------------------------
 */
console.log("Starting server...");
dotenv.config();

const DEBUG: boolean = process.env.DEBUG === "true";

const app: Express = express();
const port = process.env.PORT || 3000;
app.use(express.json());

await connectToDatabase();

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
app.listen(port, () => {
  if (DEBUG) console.log(`Listening at http://localhost:${port}/`);
});
