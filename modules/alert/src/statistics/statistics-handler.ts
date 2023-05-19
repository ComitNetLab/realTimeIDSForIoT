import { Router, Request, Response } from "express";

const statisticsRouter = Router();

statisticsRouter.get("/request-pm", async (req: Request, res: Response) => {
  res.status(200).send("This is a tests");
});

statisticsRouter.get("/atk-dashboard", async (req: Request, res: Response) => {
  res.status(200).send("This is a tests for dashboard");
});

export default statisticsRouter;
