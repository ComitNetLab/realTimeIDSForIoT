/**
 * Attack detected interface
 */
export interface IAttackDetected {
  dateTime: string;
  node: string;
  pkgInfo: string;
}

/**
 * Time analysis statistics interface
 */
export interface ITimeAnalysis {
  dateTime: string;
  numRequests: number;
}
