import mongoose, { Schema } from "mongoose";
import { ITimeAnalysis } from "../../commonTypes.js";

/**
 * Time analysis statistics schema
 */
const timeAnalysisSchema = new Schema({
  // ISO format
  dateTime: { type: String, required: true },
  numRequests: { type: Number, required: true },
});

/**
 * Time analysis statistics model
 */
export const TimeAnalysis = mongoose.model("TimeAnalysis", timeAnalysisSchema);

/**
 * ------------------------------------------------------------
 * Entity methods
 * ------------------------------------------------------------
 */

/**
 * Create a new time analysis statistics
 * @param {ITimeAnalysis} payload - The time analysis statistics creation payload
 */
export const createTimeAnalysis = async (
  payload: ITimeAnalysis
): Promise<void> => {
  await TimeAnalysis.create(new TimeAnalysis(payload));
};

/**
 * Update the actual time Analysis statistics
 * @param {string} dateTime ISO format
 */
export const updateTimeAnalysis = async (dateTime: String): Promise<void> => {
  await TimeAnalysis.updateOne({ dateTime }, { $inc: { numRequests: 1 } });
};

/**
 * Get the last time analysis statistics
 * @returns {Promise<ITimeAnalysis | null>} The last time analysis statistics
 */
export const getLastTimeAnalysis = async (): Promise<ITimeAnalysis | null> => {
  const temp = await TimeAnalysis.find().sort({ _id: -1 }).limit(1);
  return temp.length ? temp[0] : null;
};
