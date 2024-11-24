import { useState } from "react";
import { useUploadAndProcessMutation } from "../../api";

export const ProcessingDataset = () => {
  const [uploadAndProcess] = useUploadAndProcessMutation();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (selectedFile) {
      const response = await uploadAndProcess(selectedFile).unwrap();
      alert(response.message);
    }
  };

  return (
    <div>
      <h1>Обработка датасета</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleUpload} disabled={!selectedFile}>
        Обработать
      </button>
    </div>
  );
};
