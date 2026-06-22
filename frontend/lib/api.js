// frontend/lib/api.js

const API_BASE = 'http://localhost:8000/api/v1';

// Generic API request helper.
// Automatically attaches the JWT token (if present) and redirects
// to login if the backend says the token is no longer valid.
export async function apiRequest(endpoint, options = {}) {
  const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;

  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    if (response.status === 401) {
      // Token expired or invalid - send the user back to login
      if (typeof window !== 'undefined') {
        localStorage.removeItem('access_token');
        window.location.href = '/auth/login';
      }
    }

    // Try to get the backend's error detail, fall back to status text
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch {
      // body wasn't JSON, just use statusText
    }
    throw new Error(detail);
  }

  return response.json();
}

// Convenience wrappers for the most common HTTP verbs.
// These just call apiRequest with the right method set.

export function apiGet(endpoint) {
  return apiRequest(endpoint, { method: 'GET' });
}

export function apiPost(endpoint, body) {
  return apiRequest(endpoint, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function apiDelete(endpoint) {
  return apiRequest(endpoint, { method: 'DELETE' });
}

// File uploads need FormData, not JSON - so this skips the
// Content-Type: application/json header and lets the browser
// set the correct multipart boundary automatically.
export async function apiUpload(endpoint, file) {
  const token = typeof window !== 'undefined' ? localStorage.getItem('access_token') : null;

  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  });

  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.detail || 'Upload failed');
  }

  return response.json();
}