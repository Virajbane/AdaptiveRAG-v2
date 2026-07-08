// frontend/components/document/FileUpload.jsx
'use client';

import { useState } from 'react';
import { useAuth } from '@/app/context/AuthContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// This component shows a drag-and-drop box for uploading files.
// It calls onUploadSuccess() when the upload finishes, so the parent
// page knows to refresh its document list.
export function FileUpload({ onUploadSuccess }) {
  const { token } = useAuth(); // JWT token from login, needed to prove who we are

  // Plain useState, no type annotations needed in JS
  const [isDragging, setIsDragging] = useState(false);   // true while user drags a file over the box
  const [isUploading, setIsUploading] = useState(false); // true while upload is in progress
  const [error, setError] = useState('');                // holds any error message to show
  const [progress, setProgress] = useState(0);            // fake progress bar (0-100)

  // Called continuously while a file is being dragged over the drop zone
  const handleDragOver = (e) => {
    e.preventDefault(); // stops the browser from opening the file directly
    setIsDragging(true);
  };

  // Called when the dragged file leaves the drop zone without being dropped
  const handleDragLeave = () => {
    setIsDragging(false);
  };

  // The actual upload logic - sends the file to our FastAPI backend
  const uploadFile = async (file) => {
    setError('');
    setIsUploading(true);
    setProgress(0);

    try {
      // FormData is the standard way browsers send files over HTTP
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_URL}/api/v1/documents/upload`, {
        method: 'POST',
        headers: {
          // Note: we do NOT set Content-Type here.
          // The browser sets it automatically for FormData (multipart/form-data)
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Upload failed');
      }

      const data = await response.json();
      setProgress(100);

      // Wait a moment so the user sees "100%" before we reset the UI
      setTimeout(() => {
        setIsUploading(false);
        onUploadSuccess(); // tell the parent page to refresh the document list
      }, 1000);
    } catch (err) {
      // err.message works the same in JS as TS, we just don't need "instanceof Error" checks
      setError(err.message || 'Upload failed');
      setIsUploading(false);
    }
  };

  // Called when user drops a file onto the box
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      uploadFile(files[0]); // we only handle the first file
    }
  };

  // Called when user clicks the box and picks a file via the OS file picker
  const handleFileSelect = (e) => {
    const files = e.currentTarget.files;
    if (files && files.length > 0) {
      uploadFile(files[0]);
    }
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`border-2 border-dashed rounded-lg p-8 text-center transition ${
        isDragging
          ? 'border-blue-500 bg-blue-50'
          : 'border-gray-300 bg-gray-50'
      } ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}
    >
      {/* Hidden native file input - the visible UI is the label below */}
      <input
        type="file"
        id="file-input"
        onChange={handleFileSelect}
        accept=".pdf,.docx,.txt,.csv"
        className="hidden"
        disabled={isUploading}
      />

      {!isUploading ? (
        // Default state: prompt to drag or click
        <label htmlFor="file-input" className="cursor-pointer">
          <div className="text-4xl mb-2">📄</div>
          <p className="text-lg font-semibold text-gray-900">
            Drag and drop your file here
          </p>
          <p className="text-sm text-gray-500 mt-2">
            or click to select (PDF, DOCX, TXT, CSV up to 50MB)
          </p>
        </label>
      ) : (
        // Uploading state: show a progress bar
        <div>
          <p className="text-lg font-semibold mb-4">Uploading...</p>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {error && <p className="text-red-600 text-sm mt-4">{error}</p>}
    </div>
  );
}