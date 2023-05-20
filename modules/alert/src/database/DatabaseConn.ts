import mongoose, { Mongoose } from "mongoose";

export const connectToDatabase = async () => {
  console.log("Creating DatabaseConn");
  const addLogin = process.env.MONGO_USER && process.env.MONGO_PASSWORD;
  let mongoURL = "";
  if (process.env.MONGO_HOST && process.env.MONGO_PORT)
    mongoURL = addLogin
      ? `mongodb://${process.env.MONGO_USER}:${process.env.MONGO_PASSWORD}@${process.env.MONGO_HOST}:${process.env.MONGO_PORT}`
      : `mongodb://${process.env.MONGO_HOST}:${process.env.MONGO_PORT}/alerts`;
  else mongoURL = "mongodb://localhost:2701/alerts";
  console.log(`Connecting to ${mongoURL}`);
  await mongoose.connect(mongoURL, { dbName: "alerts" });
  console.log("Connected to MongoDB");
};
