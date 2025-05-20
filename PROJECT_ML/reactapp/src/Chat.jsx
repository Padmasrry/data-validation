import React, { useState, useEffect } from 'react';
import './Chat.css';

const Chat = ({ user }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [uniqueId, setUniqueId] = useState('');
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState('');
  const [empId] = useState(user ? user.empId : 'padmasrry');

  useEffect(() => {
    if (isOpen && messages.length === 0) {
      const defaultMessage = { text: "What can I help you with?", sender: 'bot', timestamp: new Date().toLocaleTimeString() };
      setMessages([defaultMessage]);
    }
    setQuestion(''); // Clear question after response
  }, [isOpen, messages.length]);

  const handleChatRequest = async () => {
    if (!uniqueId && !question.toLowerCase().includes('files') && !question.toLowerCase().includes('how many')) {
      setError('Please enter a unique_id first or ask about files/how many files!');
      return;
    }
    if (!question.trim()) {
      setError('Please enter a question!');
      return;
    }
    try {
      const res = await fetch('http://127.0.0.1:8000/chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'empId': empId,
        },
        body: JSON.stringify({ question: question, file_id: uniqueId || '' }),
      });
      const data = await res.json();
      setMessages(prevMessages => [
        ...prevMessages,
        { text: question, sender: 'user', timestamp: new Date().toLocaleTimeString() },
        { text: data.response || 'No response', sender: 'bot', timestamp: new Date().toLocaleTimeString() }
      ]);
      setError('');
      setQuestion('');
    } catch (err) {
      setError('Failed to connect to chatbot. Try again.');
      setMessages(prevMessages => [
        ...prevMessages,
        { text: question, sender: 'user', timestamp: new Date().toLocaleTimeString() },
        { text: error, sender: 'bot', timestamp: new Date().toLocaleTimeString() }
      ]);
      setQuestion('');
    }
  };

  const handleIdSubmit = () => {
    if (uniqueId && !isNaN(uniqueId)) {
      setError('');
    } else {
      setError('Please enter a valid unique_id (numbers only)!');
    }
  };

  if (!isOpen) {
    return (
      <div
        className="chat-toggle"
        onClick={() => setIsOpen(true)}
        style={{ position: 'fixed', bottom: '20px', right: '20px', cursor: 'pointer', zIndex: 1000 }}
      >
        <img src="/chatbot.png" alt="Chat" style={{ width: '40px' }} />
      </div>
    );
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h3>Virtual Assistant</h3>
        <button onClick={() => setIsOpen(false)}>X</button>
      </div>
      {!uniqueId ? (
        <>
          <input
            type="text"
            placeholder="Enter unique_id (e.g., 1, 4, 5)"
            value={uniqueId}
            onChange={(e) => setUniqueId(e.target.value)}
            className="chat-input"
          />
          <button onClick={handleIdSubmit} className="chat-button">Submit ID</button>
          {error && <p className="chat-error">{error}</p>}
        </>
      ) : (
        <>
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}`}>
                <p>{msg.text}</p>
                <span className="timestamp">{msg.timestamp}</span>
              </div>
            ))}
          </div>
          <div className="chat-input-container">
            <input
              type="text"
              placeholder="Ask a question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="chat-input"
            />
            <button onClick={handleChatRequest} className="chat-button">Ask</button>
          </div>
          {error && <p className="chat-error">{error}</p>}
        </>
      )}
    </div>
  );
};

export default Chat;