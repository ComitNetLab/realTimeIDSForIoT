import mongoose, { Schema } from "mongoose";
import { IAttackDetected } from "../../commonTypes.js";

const attackDetectedSchema = new Schema({
  dateTime: { type: String, required: true },
  node: { type: String, required: true, enum: [] },
  pkgInfo: { type: String, required: true },
});

export const AttackDetected = mongoose.model(
  "AttackDetected",
  attackDetectedSchema
);

export const createAttackDetected = async (payload: IAttackDetected) => {
  const ad = new AttackDetected(payload);
  return await ad.save();
};
