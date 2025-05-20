import React, { useState } from 'react';
import axios from 'axios';
import './FileUpload.css';

const FileUpload = ({ initialFile, initialStatus = '', acceptTypes = '.csv', onUpload }) => {
  const [file, setFile] = useState(initialFile || null);
  const [status, setStatus] = useState(initialStatus);
  const [anomalies, setAnomalies] = useState(null); // Add state for anomalies

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      alert("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setStatus("Validating...");
      if (onUpload) {
        // If onUpload prop is provided (e.g., in Storybook), use it instead of axios
        const response = await onUpload(formData);
        setStatus(response.message);
        setAnomalies(response.anomalies || 0); // Set anomalies from response
      } else {
        // Default behavior for the app
        const response = await axios.post("http://127.0.0.1:8000/upload_csv/", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });
        setStatus(response.data.message);
        setAnomalies(response.data.anomalies || 0); // Set anomalies from response
      }
    } catch (error) {
      setStatus("Error during validation.");
      setAnomalies(null);
      console.error("Error:", error);
    }
  };

  return (
    <div>
      <h1>CSV File Upload and Validation</h1>
      <form onSubmit={handleSubmit}>
        <input type="file" accept={acceptTypes} onChange={handleFileChange} />
        <button type="submit">Upload and Validate</button>
      </form>
      <p>Status: {status}</p>
      {anomalies !== null && <p>Anomalies Detected: {anomalies}</p>} {/* Add anomalies display */}
    </div>
  );
};

export default FileUpload;