import React, { useState } from 'react';
import './Register.css';
import { User, Calendar, Building, Briefcase, Mail, Phone } from 'lucide-react';

const Register = ({ onRegisterSuccess, setCurrentPage }) => {
  const [formData, setFormData] = useState({
    empId: '',
    name: '',
    dob: '',
    companyBranch: '',
    designation: '',
    email: '',
    phone: '',
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Validate DOB format
    if (!/^\d{4}-\d{2}-\d{2}$/.test(formData.dob)) {
      alert('DOB must be in YYYY-MM-DD format');
      return;
    }
    fetch('http://127.0.0.1:8000/register/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          alert('Registration successful! Please log in.');
          onRegisterSuccess(); // Navigate to login page
        } else {
          alert('Registration failed: ' + data.message);
        }
      })
      .catch((error) => {
        console.error('Registration error:', error);
        alert('Registration failed');
      });
  };

  return (
    <div className="register-container">
      <div className="register-left">
        <h1>Data Validation</h1>
        <p>
          Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
        </p>
        <div className="decorative-shapes"></div>
      </div>
      <div className="register-right">
        <h2>Register</h2>
        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <User className="input-icon" />
            <input
              type="text"
              name="empId"
              placeholder="Employee ID"
              value={formData.empId}
              onChange={handleChange}
              required
            />
          </div>
          <div className="input-group">
            <User className="input-icon" />
            <input
              type="text"
              name="name"
              placeholder="Name"
              value={formData.name}
              onChange={handleChange}
              required
            />
          </div>
          <div className="input-group">
            <Calendar className="input-icon" />
            <input
              type="date"
              name="dob"
              value={formData.dob}
              onChange={handleChange}
              required
            />
          </div>
          <div className="input-group">
            <Building className="input-icon" />
            <input
              type="text"
              name="companyBranch"
              placeholder="Company Branch"
              value={formData.companyBranch}
              onChange={handleChange}
              required
            />
          </div>
          <div className="input-group">
            <Briefcase className="input-icon" />
            <input
              type="text"
              name="designation"
              placeholder="Designation"
              value={formData.designation}
              onChange={handleChange}
              required
            />
          </div>
          <div className="input-group">
            <Mail className="input-icon" />
            <input
              type="email"
              name="email"
              placeholder="Email ID"
              value={formData.email}
              onChange={handleChange}
              required
            />
          </div>
          <div className="input-group">
            <Phone className="input-icon" />
            <input
              type="tel"
              name="phone"
              placeholder="Phone Number"
              value={formData.phone}
              onChange={handleChange}
              required
            />
          </div>
          <button type="submit" className="register-btn">Submit</button>
        </form>
      </div>
    </div>
  );
};

export default Register;