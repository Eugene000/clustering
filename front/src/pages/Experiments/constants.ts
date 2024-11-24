export const architectures = [
  { id: "arch_1", value: "64,32,16", label: "3-х слойная (64, 32, 16)" },
  { id: "arch_2", value: "64,32,16,8", label: "4-х слойная (64, 32, 16, 8)" },
  { id: "arch_3", value: "64,16", label: "2-х слойная (64, 16)" },
];

export const latentDims = [
  { id: "ld_1", value: 3, label: "3" },
  { id: "ld_2", value: 5, label: "5" },
];

export const epochs = [
  { id: "ep_1", value: 50, label: "50" },
  { id: "ep_2", value: 20, label: "20" },
  { id: "ep_3", value: 100, label: "100" },
];

export const batchSizes = [
  { id: "bs_1", value: 32, label: "32" },
  { id: "bs_2", value: 64, label: "64" },
  { id: "bs_3", value: 16, label: "16" },
];

export const learningRates = [
  { id: "lr_1", value: 0.001, label: "0.001" },
  { id: "lr_2", value: 0.005, label: "0.005" },
  { id: "lr_3", value: 0.01, label: "0.01" },
];

export const activations = [
  { id: "act_1", value: "relu", label: "ReLU" },
  { id: "act_2", value: "tanh", label: "Tanh" },
];

export const options = {
  architectures,
  latentDims,
  epochs,
  batchSizes,
  learningRates,
  activations,
};

export enum PARAM_KEYS {
  ARCHITECTURE = "architecture",
  LATENT_DIM = "latentDim",
  EPOCH = "epoch",
  BATCH_SIZE = "batchSize",
  LEARNING_RATE = "learningRate",
  ACTIVATION = "activation",
}

export const parameters = [
  { label: "Архитектура", key: PARAM_KEYS.ARCHITECTURE, options: options.architectures },
  {
    label: "Размер латентного пространства",
    key: PARAM_KEYS.LATENT_DIM,
    options: options.latentDims,
  },
  { label: "Количество эпох", key: PARAM_KEYS.EPOCH, options: options.epochs },
  { label: "Размер батча", key: PARAM_KEYS.BATCH_SIZE, options: options.batchSizes },
  { label: "Скорость обучения", key: PARAM_KEYS.LEARNING_RATE, options: options.learningRates },
  { label: "Функция активации", key: PARAM_KEYS.ACTIVATION, options: options.activations },
];
