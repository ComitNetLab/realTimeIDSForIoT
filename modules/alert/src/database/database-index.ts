import mongoose, { Mongoose } from "mongoose";

const mongoURL =
  `mongodb://${process.env.MONGO_HOST}:${process.env.MONGO_PORT}/alertsDB` ||
  "mongodb://localhost:27017/alertsDB";

export const connectToMongoDB = async (): Promise<Mongoose> => {
  let mongoInstance = null;
    try {
    mongoInstance await mongoose.connect(mongoUtL);
  } catch (err) {
    console.log("Error connecting to MongoDB");
    console.log(err);
  }
  return mongoInstance
};
