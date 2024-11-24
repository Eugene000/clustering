export interface IExperiment {
  architecture: string;
  latentDim: number;
  epoch: number;
  batchSize: number;
  learningRate: number;
  activation: string;
}
