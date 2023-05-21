import mongoose, { Schema } from "mongoose";
import { IAttackDetected } from "../../commonTypes.js";

/**
 * Attack detected schema
 */
const attackDetectedSchema = new Schema({
  dateTime: { type: String, required: true },
  node: { type: String, required: true, enum: [] },
  pkgInfo: { type: String, required: true },
});

/**
 * Attack detected model
 */
export const AttackDetected = mongoose.model(
  "AttackDetected",
  attackDetectedSchema
);

/**
 * ------------------------------------------------------------
 * Entity methods
 * ------------------------------------------------------------
 */

/**
 * Create a new attack detected
 * @param {IAttackDetected} payload - The attack detected creation payload
 */
export const createAttackDetected = async (
  payload: IAttackDetected
): Promise<void> => {
  await AttackDetected.create(new AttackDetected(payload));
};

/**
 * Get the last 10 attacks detected for the dashboard
 * @returns {Promise<IAttackDetected[]>} The last 10 attacks detected
 */
export const getLastAttacksDetected = async (): Promise<IAttackDetected[]> => {
  return (await AttackDetected.find().sort({ _id: -1 }).limit(10)).map(
    (ad) => ({ node: ad.node, pkgInfo: ad.pkgInfo, dateTime: ad.dateTime })
  );
};
