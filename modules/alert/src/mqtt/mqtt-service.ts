import { createAttackDetected } from "../database/models/AttackDetected.js";
import {
  createTimeAnalysis,
  getLastTimeAnalysis,
  updateTimeAnalysis,
} from "../database/models/TimeAnalysis.js";

const processAttackDetected = async (msg: string[]): Promise<void> => {
  const node = msg[0];
  const pkgInfo = msg[1];
  await createAttackDetected({
    dateTime: new Date().toISOString(),
    node,
    pkgInfo,
  });
};

const processAttackDetection = async (): Promise<void> => {
  const lastTimeAnalysis = await getLastTimeAnalysis();
  const actualTime = new Date();
  actualTime.setSeconds(0);
  actualTime.setMilliseconds(0);
  if (lastTimeAnalysis) {
    console.log(lastTimeAnalysis.dateTime, actualTime.toISOString());
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
export const onMessage = async (
  topic: string,
  message: string
): Promise<void> => {
  if (topic === "atkDetection") await processAttackDetection();
  else if (topic === "atkDetected") {
    const msg = message.toString().split(";");
    await processAttackDetected(msg);
  }
};
