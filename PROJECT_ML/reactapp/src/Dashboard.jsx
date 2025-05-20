import React, { useEffect, useState } from 'react';
import { Doughnut, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { File, Clock, Activity } from 'lucide-react';
import './Dashboard.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard = (props) => {
  const [metrics, setMetrics] = useState(props || {
    totalFiles: 0,
    averageMonthlyRuns: 0,
    recentActivity: [], // Default to empty array
    monthlyBreakdown: [],
  });
  const [timePeriod, setTimePeriod] = useState(props?.timePeriod || '30');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!props?.totalFiles) {
      setLoading(true);
      const timeoutId = setTimeout(() => setLoading(false), 10000); // Timeout after 10s
      fetch(`http://127.0.0.1:8000/get_dashboard_metrics/?period=${timePeriod}`)
        .then((response) => {
          if (!response.ok) throw new Error("Network response was not ok");
          return response.json();
        })
        .then((data) => {
          console.log("Dashboard data:", data);
          setMetrics({
            totalFiles: data.total_files || 0,
            averageMonthlyRuns: data.average_monthly_runs || 0,
            recentActivity: data.recent_activity || [], // Ensure default empty array
            monthlyBreakdown: data.monthly_breakdown || [],
          });
        })
        .catch((error) => {
          console.error("Error fetching dashboard metrics:", error);
          setError("Failed to load data. Check backend or try again.");
        })
        .finally(() => {
          clearTimeout(timeoutId);
          setLoading(false);
        });
    } else {
      setMetrics(props);
      setLoading(false);
    }
  }, [timePeriod, props]);

  if (loading) {
    return <p className="loading-text">Loading...</p>;
  }

  if (error) {
    return <p className="error-text">{error}</p>;
  }

  // Data for existing charts
  const today = new Date().toLocaleDateString();
  const todayActivity = metrics.recentActivity.filter(activity => {
    const activityDate = new Date(activity.split(' on ')[1].split(' with ')[0]).toLocaleDateString();
    return activityDate === today;
  });
  const uploadCount = todayActivity.length || 1;
  const progress = Math.min(100, (uploadCount / 10) * 100);
  const tooltipContent = `${today}: ${uploadCount} upload${uploadCount !== 1 ? 's' : ''} today`;

  const recentActivityChartData = {
    labels: ['Uploads', 'Remaining'],
    datasets: [{
      data: [progress, 100 - progress],
      backgroundColor: ['#28a745', '#e9ecef'],
      hoverOffset: 4,
    }],
  };

  const recentActivityChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          label: () => tooltipContent,
        },
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        borderColor: '#fff',
        borderWidth: 1,
        padding: 10,
        bodyFont: { size: 12 },
      },
    },
    cutout: '70%',
  };

  const averageRunsChartData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
    datasets: [{
      label: 'Runs',
      data: Array(5).fill(metrics.averageMonthlyRuns / 5),
      backgroundColor: '#6c757d',
      borderColor: '#6c757d',
      borderWidth: 1,
    }, {
      label: 'Previous',
      data: Array(5).fill((metrics.averageMonthlyRuns / 5) * 0.8),
      backgroundColor: '#dc3545',
      borderColor: '#dc3545',
      borderWidth: 1,
    }],
  };

  const averageRunsChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { display: false }, y: { display: false } },
    barThickness: 10,
  };

  // Data for monthly bar graph with anomalies
  const monthlyBarChartData = {
    labels: metrics.monthlyBreakdown.map(item => item.month),
    datasets: [
      {
        label: 'Total Files Uploaded',
        data: metrics.monthlyBreakdown.map(item => item.file_count),
        backgroundColor: [
          '#a3e4d7', '#96ceb4', '#88d8b0', '#7accc2', '#6dc0d5', '#5fb4e8', '#52a9eb', '#459dee', '#3792f0', '#2a87f3', '#1d7cf6', '#1071f9',
        ],
        borderColor: '#fff',
        borderWidth: 1,
      },
      {
        label: 'Total Anomalies',
        data: metrics.monthlyBreakdown.map(item => item.total_anomalies || 0),
        backgroundColor: [
          '#ff9999', '#ff8080', '#ff6666', '#ff4d4d', '#ff3333', '#ff1a1a', '#ff0000', '#e60000', '#cc0000', '#b30000', '#990000', '#800000',
        ],
        borderColor: '#fff',
        borderWidth: 1,
      },
    ],
  };

  const monthlyBarChartOptions = {
    indexAxis: 'y', // Force horizontal bar graph
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top' },
      tooltip: {
        callbacks: {
          label: (tooltipItem) => {
            const month = tooltipItem.label;
            const datasetLabel = tooltipItem.dataset.label;
            const value = tooltipItem.raw;
            const monthData = metrics.monthlyBreakdown.find(item => item.month === month);
            const totalFiles = monthData ? monthData.file_count : 0;
            const avgRuns = monthData ? monthData.average_runs : 0;
            const totalAnomalies = monthData ? monthData.total_anomalies || 0 : 0;
            if (datasetLabel === 'Total Files Uploaded') {
              return [
                `Total Files Uploaded: ${totalFiles}`,
                `Average Runs: ${avgRuns.toFixed(2)}`,
                `Total Anomalies: ${totalAnomalies}`,
              ];
            }
            return `${datasetLabel}: ${value}`;
          },
        },
        backgroundColor: 'rgba(0, 0, 0, 0.9)',
        borderColor: '#fff',
        borderWidth: 1,
        padding: 10,
        bodyFont: { size: 12 },
      },
    },
    scales: {
      x: {
        title: { display: true, text: 'Count' },
        beginAtZero: true,
      },
      y: {
        title: { display: true, text: 'Month' },
        ticks: { autoSkip: false },
      },
    },
  };

  return (
    <div className="dashboard">
      <h1>Dashboard</h1>
      <div className="dashboard-stats">
        <div className="stat-card blue">
          <File className="stat-icon" />
          <h3>Total Files Uploaded</h3>
          <p>{metrics.totalFiles}</p>
          <span className="stat-change">+22% <span className="arrow-up">↑</span></span>
          <button className="period-btn" onClick={() => setTimePeriod('30')}>Last 30 Days</button>
        </div>
        <div className="stat-card gray">
          <Clock className="stat-icon" />
          <h3>Average Runs</h3>
          <p>{metrics.averageMonthlyRuns.toFixed(2)}</p>
          <span className="stat-change">+38% <span className="arrow-up">↑</span></span>
          <div className="chart-container">
            <Bar data={averageRunsChartData} options={averageRunsChartOptions} />
          </div>
          <button className="period-btn" onClick={() => setTimePeriod('30')}>Last 30 Days</button>
        </div>
        <div className="stat-card green">
          <Activity className="stat-icon" />
          <h3>Recent Activity</h3>
          <div className="chart-container">
            <Doughnut data={recentActivityChartData} options={recentActivityChartOptions} />
          </div>
          <p>{progress}%</p>
          <button className="period-btn" onClick={() => setTimePeriod('1')}>Today</button>
        </div>
      </div>
      {/* Monthly bar graph with anomalies */}
      <div className="monthly-bar-graph-container">
        <h3>Monthly File Upload Rate</h3>
        <div className="chart-container scrollable">
          <Bar data={monthlyBarChartData} options={monthlyBarChartOptions} />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;