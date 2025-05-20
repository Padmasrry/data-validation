import React, { useEffect, useState } from "react";
import "./ValidationLogs.css";

const ValidationLogs = ({ initialLogs = [], initialLoading = true, initialError = null, mockFetch, empId }) => {
  const [logs, setLogs] = useState(initialLogs);
  const [loading, setLoading] = useState(initialLoading);
  const [error, setError] = useState(initialError);

  useEffect(() => {
    if (mockFetch) {
      // Use mockFetch prop for Storybook testing
      mockFetch()
        .then((response) => response.json())
        .then((data) => {
          console.log(data);
          setLogs(data);
          setLoading(false);
        })
        .catch((error) => {
          console.error("Error fetching logs:", error);
          setError("Failed to fetch logs. Please try again later.");
          setLoading(false);
        });
    } else {
      // Default behavior for the app with empId in headers
      setLoading(true);
      const timeoutId = setTimeout(() => setLoading(false), 10000); // Timeout after 10s
      fetch("http://127.0.0.1:8000/get_validation_logs/", {
        headers: { 'empId': empId || 'padmasrry' }
      })
        .then((response) => {
          if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
          return response.json();
        })
        .then((data) => {
          console.log(data);
          setLogs(Array.isArray(data) ? data : []);
        })
        .catch((error) => {
          console.error("Error fetching logs:", error);
          setError("Failed to fetch logs. Please try again later.");
        })
        .finally(() => {
          clearTimeout(timeoutId);
          setLoading(false);
        });
    }
  }, [mockFetch, empId]);

  return (
    <div className="log-container">
      <h2 className="log-title">Validation Logs</h2>
      {loading ? (
        <p className="loading-text">Loading...</p>
      ) : error ? (
        <p className="error-text">{error}</p>
      ) : logs.length === 0 ? (
        <p>No logs available.</p>
      ) : (
        <table className="log-table">
          <thead>
            <tr>
              <th>File Name</th>
              <th>Total Errors</th>
              <th>Time Taken (s)</th>
              <th>Start Time</th>
              <th>End Time</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log, index) => (
              <tr key={index}>
                <td>{log.file_name}</td>
                <td>{log.total_errors}</td>
                <td>{log.time_taken}</td>
                <td>{new Date(log.start_time).toLocaleString()}</td>
                <td>{new Date(log.end_time).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default ValidationLogs;