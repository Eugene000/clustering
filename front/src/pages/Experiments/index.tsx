import React, { useState } from "react";
import { options, PARAM_KEYS, parameters } from "./constants";
import { useExperimentMutation } from "../../api";
import { IExperiment } from "../../api/types";

export const Experiments = () => {
  const [executeExperiment] = useExperimentMutation();

  const [formData, setFormData] = useState<IExperiment>({
    architecture: options.architectures[0].value,
    latentDim: options.latentDims[0].value,
    epoch: options.epochs[0].value,
    batchSize: options.batchSizes[0].value,
    learningRate: options.learningRates[0].value,
    activation: options.activations[0].value,
  });

  const handleChange = (key: PARAM_KEYS, value: string | number) => {
    setFormData((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const response = await executeExperiment(formData).unwrap();
    alert(`${response.silhouette}, ${response.davies_bouldin}`);
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>Проведение экспериментов</h2>
      <form onSubmit={handleSubmit}>
        {parameters.map(({ label, key, options }) => (
          <div key={key} style={{ marginBottom: "10px" }}>
            <label>
              {label}:
              <select value={formData[key]} onChange={(e) => handleChange(key, e.target.value)}>
                {options.map((option) => (
                  <option key={option.id} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
        ))}

        <div style={{ marginTop: "20px" }}>
          <button type="submit">Submit Experiment</button>
        </div>
      </form>
    </div>
  );
};
