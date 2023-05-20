export interface IAttackDetected {
  dateTime: string;
  node: string;
  pkgInfo: string;
}

export interface ITimeAnalysis {
  dateTime: string;
  numRequests: number;
}
