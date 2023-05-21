import { createAttackDetected } from "../database/models/AttackDetected.js";
import {
  createTimeAnalysis,
  getLastTimeAnalysis,
  updateTimeAnalysis,
} from "../database/models/TimeAnalysis.js";

/**
 * Process the attack detected message.
 * @param {string[]} msg - The message received from the MQTT broker
 */
const processAttackDetected = async (msg: string[]): Promise<void> => {
  const node = msg[0];
  const pkgInfo = msg[1];
  await createAttackDetected({
    dateTime: new Date().toISOString(),
    node,
    pkgInfo,
  });
};

/**
 * Process the attack detection message.
 * If the last time analysis is not the actual time, create a new time analysis.
 * If the last time analysis is the actual time, update the number of requests.
 */
const processAttackDetection = async (): Promise<void> => {
  const lastTimeAnalysis = await getLastTimeAnalysis();
  const actualTime = new Date();
  actualTime.setSeconds(0);
  actualTime.setMilliseconds(0);
  if (lastTimeAnalysis) {
    if (lastTimeAnalysis.dateTime !== actualTime.toISOString()) {
      await createTimeAnalysis({
        dateTime: actualTime.toISOString(),
        numRequests: 1,
      });
    } else {
      updateTimeAnalysis(actualTime.toISOString());
    }
  } else {
    await createTimeAnalysis({
      dateTime: actualTime.toISOString(),
      numRequests: 1,
    });
  }
};

/**
 * Function to handle the MQTT messages
 * @param {string} topic - The topic of the message
 * @param {Buffer} message - The received message buffer
 */
export const onMessage = async (
  topic: string,
  message: Buffer
): Promise<void> => {
  if (topic === "atkDetection") await processAttackDetection();
  else if (topic === "atkDetected") {
    // Message format: node;pkgInfo
    // pkgInfo format is separated by "," and the content varies depending on the node
    const msg = message.toString().split(";");
    await processAttackDetected(msg);
  }
};
