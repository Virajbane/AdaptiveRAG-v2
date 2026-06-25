import axios from "axios";
import { isTokenExpired } from "../config/security";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { "Content-Type": "application/json" },
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) {
    if (isTokenExpired(token)) {
      localStorage.removeItem("access_token");
      localStorage.removeItem("user_id");
      localStorage.removeItem("user_name");
      window.location.href = "/auth/login?session_expired=true";
      return Promise.reject(new Error("Token expired"));
    }
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem("access_token");
      localStorage.removeItem("user_id");
      localStorage.removeItem("user_name");
      window.location.href = "/auth/login?session_expired=true";
    }
    return Promise.reject(error);
  }
);