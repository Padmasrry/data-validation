import React, { useState, useEffect } from 'react';
import './HomePage.css';
import FileUpload from './FileUpload';
import ValidationLogs from './ValidationLogs';
import CSVDownload from './CSVDownload';
import Dashboard from './Dashboard';
import Login from './Login';
import Register from './Register';
import Chat from './Chat';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faComment } from '@fortawesome/free-solid-svg-icons';

export default function HomePage() {
  const [currentPage, setCurrentPage] = useState('login');
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState(null);
  const [isChatOpen, setIsChatOpen] = useState(false);

  useEffect(() => {
    const storedLogin = localStorage.getItem('isLoggedIn');
    if (storedLogin === 'true') {
      setIsLoggedIn(true);
      setCurrentPage('dashboard');
    }
  }, []);

  const handleLoginSuccess = (username, password) => {
    fetch('http://127.0.0.1:8000/login/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          setIsLoggedIn(true);
          setUser({ empId: data.empId });
          console.log('Logged in with empId:', data.empId); // Debug the empId
          localStorage.setItem('isLoggedIn', 'true');
          setCurrentPage('dashboard');
        } else {
          alert('Invalid credentials');
        }
      })
      .catch((error) => {
        console.error('Login error:', error);
        alert('Login failed');
      });
  };

  const handleRegisterSuccess = () => {
    setCurrentPage('login');
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setUser(null);
    localStorage.removeItem('isLoggedIn');
    setCurrentPage('login');
  };

  return (
    <div className={`container ${isLoggedIn ? '' : 'login-mode'}`}>
      {isLoggedIn && (
        <nav className="sidebar">
          <div className="sidebar-title">Home</div>
          <button className="nav-link" onClick={() => setCurrentPage('dashboard')}>
            Dashboard
          </button>
          <button className="nav-link" onClick={() => setCurrentPage('input')}>
            Input File
          </button>
          <button className="nav-link" onClick={() => setCurrentPage('status')}>
            Run Status
          </button>
          <button className="nav-link" onClick={() => setCurrentPage('reports')}>
            Reports
          </button>
          <button className="nav-link" onClick={handleLogout}>
            Logout
          </button>
        </nav>
      )}
      <div className="main-content">
        {currentPage === 'login' && <Login onLoginSuccess={handleLoginSuccess} setCurrentPage={setCurrentPage} />}
        {currentPage === 'register' && <Register onRegisterSuccess={handleRegisterSuccess} setCurrentPage={setCurrentPage} />}
        {isLoggedIn && currentPage === 'dashboard' && <Dashboard />}
        {isLoggedIn && currentPage === 'input' && <FileUpload />}
        {isLoggedIn && currentPage === 'status' && <ValidationLogs empId={user ? user.empId : 'padmasrry'} />}
        {isLoggedIn && currentPage === 'reports' && (
          <CSVDownload
            fetchOptions={{ headers: { 'empId': user ? user.empId : 'padmasrry' } }}
          />
        )}
        {isLoggedIn && (
          <>
            <div
              className="chat-toggle"
              onClick={() => setIsChatOpen(!isChatOpen)}
              style={{ position: 'fixed', bottom: '20px', right: '20px', cursor: 'pointer', zIndex: 1000 }}
            >
              <FontAwesomeIcon icon={faComment} size="2x" color="#4CAF50" />
            </div>
            {isLoggedIn && <Chat user={user} />}
          </>
        )}
      </div>
    </div>
  );
}