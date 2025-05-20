import React, { useState } from 'react';
import './Login.css';
import { User, Lock } from 'lucide-react'; // Using lucide-react for icons

const Login = ({ onLoginSuccess, setCurrentPage }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [remember, setRemember] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    // Validate DOB format (YYYY-MM-DD)
    if (!/^\d{4}-\d{2}-\d{2}$/.test(password)) {
      alert('Password must be DOB in YYYY-MM-DD format');
      return;
    }
    onLoginSuccess(username, password); // Call the parent handler
  };

  return (
    <div className="login-container">
      <div className="login-left">
        <h1>Data Validation</h1>
        <p>
            Validating big data for retail company
        </p>
        <div className="decorative-shapes"></div>
      </div>
      <div className="login-right">
        <h2>User Login</h2>
        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <User className="input-icon" />
            <input
              type="text"
              placeholder="Employee ID or Name"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
          </div>
          <div className="input-group">
            <Lock className="input-icon" />
            <input
              type="text"
              placeholder="Date of Birth (YYYY-MM-DD)"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <div className="options">
            <label>
              <input
                type="checkbox"
                checked={remember}
                onChange={(e) => setRemember(e.target.checked)}
              />{' '}
              Remember
            </label>
            <button className="forgot-password-btn">Forgot password?</button>
          </div>
          <button type="submit" className="login-btn">Login</button>
          <button type="button" className="register-btn" onClick={() => setCurrentPage('register')}>Register</button>
        </form>
      </div>
    </div>
  );
};

export default Login;