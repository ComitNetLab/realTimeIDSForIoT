import { Router, Request, Response } from "express";
import { getLastAttacksDetected } from "../database/models/AttackDetected.js";

const statisticsRouter = Router();

statisticsRouter.get("/request-pm", async (req: Request, res: Response) => {
  res.status(200).send("This is a tests");
});

statisticsRouter.get("/atk-dashboard", async (req: Request, res: Response) => {
  const lastAttacks = await getLastAttacksDetected();
  res.status(200).send(JSON.stringify(lastAttacks, undefined, 4));
});

export default statisticsRouter;
