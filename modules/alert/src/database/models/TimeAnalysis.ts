import mongoose, { Schema } from "mongoose";
import { ITimeAnalysis } from "../../commonTypes.js";

const timeAnalysisSchema = new Schema({
  // ISO format
  dateTime: { type: String, required: true },
  numRequests: { type: Number, required: true },
});

export const TimeAnalysis = mongoose.model("TimeAnalysis", timeAnalysisSchema);

export const createTimeAnalysis = async (payload: ITimeAnalysis) => {
  const ta = new TimeAnalysis(payload);
  return await ta.save();
};
