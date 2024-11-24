import { createApi, fetchBaseQuery } from "@reduxjs/toolkit/query/react";
import { IExperiment } from "./types";

export const apiSlice = createApi({
  reducerPath: "api",
  baseQuery: fetchBaseQuery({ baseUrl: "http://127.0.0.1:8000/" }),
  tagTypes: ["Processed", "Experiment"],
  endpoints: (builder) => ({
    // Эндпоинт для загрузки и обработки файла
    uploadAndProcess: builder.mutation<{ message: string }, File>({
      query: (file) => {
        const formData = new FormData();
        formData.append("file", file);
        return {
          url: "upload-and-process/",
          method: "POST",
          body: formData,
        };
      },
      invalidatesTags: ["Processed"],
    }),
    // Эндпоинт для эксперимента с моделью
    experiment: builder.mutation<{ silhouette: number; davies_bouldin: number }, IExperiment>({
      query: (data) => {
        return {
          url: "experiment/",
          method: "POST",
          body: data,
        };
      },
      invalidatesTags: ["Experiment"],
    }),
  }),
});

export const { useUploadAndProcessMutation, useExperimentMutation } = apiSlice;
