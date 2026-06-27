'use client';

import { useState, useEffect } from 'react';
import { useAuth } from '@/app/context/AuthContext';

export default function MemoryPage() {
  const { token } = useAuth();
  const [loading, setLoading] = useState(true);
  const [messages, setMessages] = useState([]);
  const [summaries, setSummaries] = useState([]);
  const [error, setError] = useState('');
  const [sessionId, setSessionId] = useState('default_session');


  const loadMemory = async () => {
    try {
      setLoading(true);

      const res = await fetch('http://localhost:8000/api/v1/memory/load', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ session_id: sessionId })
      });

      if (!res.ok) {
        throw new Error('Failed to load memory');
      }

      const data = await res.json();
      setMessages(data.history || []);
      setSummaries(data.summaries || []);
      setError('');
    } catch (err) {
      console.error('Error loading memory:', err);
      setError('Failed to load conversation memory');
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    if (token) {
      loadMemory();
    }
  }, [token, sessionId]);

  const clearSession = async () => {
    if (!window.confirm('Clear this conversation?')) return;

    try {
      const res = await fetch(
        `http://localhost:8000/api/v1/memory/session/${sessionId}`,
        {
          method: 'DELETE',
          headers: {
            'Authorization': `Bearer ${token}`
          }
        }
      );

      if (res.ok) {
        setMessages([]);
        setSummaries([]);
      }
    } catch (err) {
      console.error('Error clearing session:', err);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Conversation Memory
          </h1>
          <p className="text-gray-600">
            View your conversation history and summaries
          </p>
        </div>

        <div className="bg-white rounded-lg shadow p-6 mb-8">
          <div className="flex items-center gap-4 mb-4">
            <label className="text-sm font-medium text-gray-700">
              Session ID:
            </label>
            <input
              type="text"
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter session ID"
            />
            <button
              onClick={loadMemory}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Load
            </button>
            <button
              onClick={clearSession}
              className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              Clear
            </button>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-6 py-4 rounded-lg mb-8">
            {error}
          </div>
        )}

        {loading && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <p className="mt-4 text-gray-600">Loading memory...</p>
          </div>
        )}

        {!loading && messages.length > 0 && (
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Conversation History ({messages.length} messages)
            </h2>
            <div className="space-y-4">
              {messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`p-4 rounded-lg ${
                    msg.role === 'user'
                      ? 'bg-blue-50 border border-blue-200'
                      : 'bg-gray-100 border border-gray-300'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-gray-900 capitalize">
                      {msg.role}
                    </span>
                    {msg.timestamp && (
                      <span className="text-sm text-gray-500">
                        {new Date(msg.timestamp).toLocaleString()}
                      </span>
                    )}
                  </div>
                  <p className="text-gray-800">{msg.content}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {!loading && messages.length === 0 && (
          <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
            <p className="text-gray-600">No conversation history yet</p>
          </div>
        )}

        {!loading && summaries.length > 0 && (
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Session Summaries
            </h2>
            <div className="space-y-4">
              {summaries.map((summary, idx) => (
                <div key={idx} className="bg-white border border-gray-200 rounded-lg p-6">
                  <h3 className="font-semibold text-gray-900 mb-2">
                    {summary.summary}
                  </h3>
                  {summary.topics && (
                    <div className="flex flex-wrap gap-2">
                      {summary.topics.map((topic, tidx) => (
                        <span
                          key={tidx}
                          className="bg-blue-100 text-blue-800 text-sm px-3 py-1 rounded-full"
                        >
                          {topic}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}