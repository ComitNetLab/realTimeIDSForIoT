import { Router, Request, Response } from "express";
import { getLastAttacksDetected } from "../database/models/AttackDetected.js";
import { getLastTimeAnalysis } from "../database/models/TimeAnalysis.js";

const statisticsRouter = Router();

statisticsRouter.get("/request-pm", async (req: Request, res: Response) => {
  const lastTimeAnalysis = await getLastTimeAnalysis();
  res.status(200).send(JSON.stringify(lastTimeAnalysis, undefined, 4));
});

statisticsRouter.get("/atk-dashboard", async (req: Request, res: Response) => {
  const lastAttacks = await getLastAttacksDetected();
  res.status(200).send(JSON.stringify(lastAttacks, undefined, 4));
});

export default statisticsRouter;
