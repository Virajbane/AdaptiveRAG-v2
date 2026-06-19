// app/context/AuthContext.jsx

'use client';

import { createContext, useContext, useEffect, useState } from 'react';

const AuthContext = createContext(undefined);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(null);
  const [userId, setUserId] = useState(null);
  const [userName, setUserName] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Load from localStorage on mount
  useEffect(() => {
    const savedToken = localStorage.getItem('access_token');
    const savedUserId = localStorage.getItem('user_id');
    const savedUserName = localStorage.getItem('user_name');

    if (savedToken && savedUserId) {
      setToken(savedToken);
      setUserId(savedUserId);
      setUserName(savedUserName || '');
    }

    setIsLoading(false);
  }, []);

  const login = (newToken, newUserId, newUserName) => {
    localStorage.setItem('access_token', newToken);
    localStorage.setItem('user_id', newUserId);
    localStorage.setItem('user_name', newUserName);

    setToken(newToken);
    setUserId(newUserId);
    setUserName(newUserName);
  };

  const logout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_id');
    localStorage.removeItem('user_name');

    setToken(null);
    setUserId(null);
    setUserName(null);
  };

  return (
    <AuthContext.Provider
      value={{
        token,
        userId,
        userName,
        login,
        logout,
        isAuthenticated: !!token,
      }}
    >
      {!isLoading && children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }

  return context;
}