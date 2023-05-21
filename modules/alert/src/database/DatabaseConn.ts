import mongoose from "mongoose";

const DEBUG = process.env.DEBUG === "true";

/**
 * Connect to the MongoDB database
 */
export const connectToDatabase = async () => {
  if (DEBUG) console.log("Creating DatabaseConn");

  // Check if login is required
  const addLogin = process.env.MONGO_USER && process.env.MONGO_PASSWORD;
  let mongoURL = "";

  // Build the connection URL
  if (process.env.MONGO_HOST && process.env.MONGO_PORT)
    mongoURL = addLogin
      ? `mongodb://${process.env.MONGO_USER}:${process.env.MONGO_PASSWORD}@${process.env.MONGO_HOST}:${process.env.MONGO_PORT}`
      : `mongodb://${process.env.MONGO_HOST}:${process.env.MONGO_PORT}/alerts`;
  else mongoURL = "mongodb://localhost:2701/alerts";

  if (DEBUG) console.log(`Connecting to ${mongoURL}`);

  await mongoose.connect(mongoURL, { dbName: "alerts" });

  if (DEBUG) console.log("Connected to MongoDB");
};
