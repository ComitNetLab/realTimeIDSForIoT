import { createTimeAnalysis } from "../database/models/TimeAnalysis.js";

export const onMessage = async (
  topic: string,
  message: string
): Promise<void> => {
  console.log(`Received message: ${message} on topic: ${topic}`);
  if (topic === "atkDetection") {
    console.log("Creating time analysis");
    const response = await createTimeAnalysis({
      dateTime: new Date().toISOString(),
      numRequests: 1,
    });
    console.log("Created time analysis", response);
  }
};
