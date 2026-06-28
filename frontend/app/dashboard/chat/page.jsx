'use client';

import { useState, useRef, useEffect } from 'react';
import { useAuth } from '@/app/context/AuthContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const CHAT_TIMEOUT_MS = 15 * 60 * 1000; // 15 minutes — matches Ollama on CPU

export default function ChatPage() {
  const { token } = useAuth();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);
  const timerRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Show elapsed time while waiting so user knows it's working
  useEffect(() => {
    if (loading) {
      setElapsedSeconds(0);
      timerRef.current = setInterval(() => {
        setElapsedSeconds((s) => s + 1);
      }, 1000);
    } else {
      clearInterval(timerRef.current);
      setElapsedSeconds(0);
    }
    return () => clearInterval(timerRef.current);
  }, [loading]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    const currentInput = input;
    setInput('');
    setLoading(true);

    // Create abort controller for this request
    abortControllerRef.current = new AbortController();
    const timeoutId = setTimeout(() => {
      abortControllerRef.current?.abort();
    }, CHAT_TIMEOUT_MS);

    try {
      const response = await fetch(`${API_URL}/api/v1/agents/chat`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: currentInput }),
        signal: abortControllerRef.current.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`${response.status}: ${errText}`);
      }

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          confidence: data.confidence,
        },
      ]);
    } catch (error) {
      clearTimeout(timeoutId);
      if (error.name === 'AbortError') {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: 'Request timed out after 15 minutes. Ollama is still processing — try a shorter question or wait.',
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: `Error: ${error.message}`,
          },
        ]);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    abortControllerRef.current?.abort();
    setLoading(false);
  };

  return (
    <div className="max-w-4xl mx-auto h-screen flex flex-col p-6">
      <div className="mb-4">
        <h1 className="text-3xl font-bold">Chat with RAG</h1>
        <p className="text-gray-600">Ask questions about your documents</p>
      </div>

      <div className="flex-1 overflow-y-auto border border-gray-300 rounded-lg p-4 mb-4 bg-gray-50">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-8">
            <p>No messages yet. Ask something about your documents!</p>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`mb-4 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-300 text-gray-900'
              }`}
            >
              <p className="whitespace-pre-wrap">{msg.content}</p>
              {msg.confidence !== undefined && (
                <p className="text-xs mt-1 opacity-70">
                  Confidence: {(msg.confidence * 100).toFixed(0)}%
                </p>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start mb-4 items-center gap-3">
            <div className="bg-gray-300 text-gray-900 px-4 py-2 rounded-lg flex items-center gap-2">
              <span className="animate-pulse">●</span>
              <span className="animate-pulse delay-100">●</span>
              <span className="animate-pulse delay-200">●</span>
              <span className="text-sm text-gray-600 ml-2">
                Thinking... {elapsedSeconds}s
              </span>
            </div>
            <button
              onClick={handleCancel}
              className="text-sm text-red-500 hover:text-red-700 underline"
            >
              Cancel
            </button>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSendMessage} className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me anything about your documents..."
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? `${elapsedSeconds}s...` : 'Send'}
        </button>
      </form>
    </div>
  );
}