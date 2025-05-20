import React, { useEffect, useState } from 'react';

const CSVDownload = ({ fetchOptions = {} }) => {
  const [logs, setLogs] = useState([]);
  const [filteredLogs, setFilteredLogs] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const timeoutId = setTimeout(() => setLoading(false), 10000); // Timeout after 10s
    fetch('http://127.0.0.1:8000/get_validation_logs/', fetchOptions)
      .then((response) => {
        if (!response.ok) throw new Error("Network response was not ok");
        return response.json();
      })
      .then((data) => {
        setLogs(data);
        setFilteredLogs(data);
      })
      .catch((error) => {
        console.error("Error fetching logs:", error);
        setError('Failed to fetch logs. Please try again later.');
      })
      .finally(() => {
        clearTimeout(timeoutId);
        setLoading(false);
      });
  }, [fetchOptions]);

  const handleSearch = (e) => {
    const query = e.target.value;
    setSearchQuery(query);

    if (query.trim() === '') {
      setFilteredLogs(logs);
    } else {
      const filtered = logs.filter((log) =>
        log.unique_id.toString().includes(query)
      );
      setFilteredLogs(filtered);
    }
  };

  return (
    <div className="log-container">
      <h2 className="log-title">Validation Logs</h2>
      <input
        type="text"
        placeholder="Search by Unique ID..."
        value={searchQuery}
        onChange={handleSearch}
        className="search-input"
      />
      {loading ? (
        <p className="loading-text">Loading...</p>
      ) : error ? (
        <p className="error-text">{error}</p>
      ) : filteredLogs.length === 0 ? (
        <p>No logs available.</p>
      ) : (
        <table className="log-table">
          <thead>
            <tr>
              <th>Unique ID</th>
              <th>File Name</th>
              <th>Total Anomalies</th>
              <th>Download Error Log</th>
            </tr>
          </thead>
          <tbody>
            {filteredLogs.map((log) => (
              <tr key={log.unique_id}>
                <td>{log.unique_id}</td>
                <td>{log.file_name}</td>
                <td>{log.total_anomalies || 0}</td>
                <td>
                  <a
                    href={`http://127.0.0.1:8000/download_csv/${log.error_log_file}`}
                    download
                    className="download-btn"
                  >
                    {log.error_log_file}
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default CSVDownload;