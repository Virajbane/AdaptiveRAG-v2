// frontend/app/dashboard/documents/page.jsx
'use client';

import { useEffect, useState } from 'react';
import { useAuth } from '@/app/context/AuthContext';
import { FileUpload } from '@/components/document/FileUpload';

// No "interface Document {...}" needed in plain JS -
// we just trust the shape of the data that comes back from the API.

export default function DocumentsPage() {
  const { token } = useAuth(); // JWT token, needed for every authenticated request

  const [documents, setDocuments] = useState([]); // list of documents from the backend
  const [loading, setLoading] = useState(true);   // true while we're fetching
  const [error, setError] = useState('');          // error message, if any

  // Fetches the user's documents from the backend.
  // We also pass this function to <FileUpload> so it can re-run
  // automatically right after a successful upload.
  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/v1/documents', {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (!response.ok) throw new Error('Failed to fetch documents');
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (err) {
      setError(err.message || 'Error loading documents');
    } finally {
      setLoading(false);
    }
  };

  // Run fetchDocuments once when the page loads, and again
  // any time the token changes (e.g. after login)
  useEffect(() => {
    fetchDocuments();
  }, [token]);

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">My Documents</h1>

      {/* Upload Section */}
      <div className="mb-12">
        <h2 className="text-xl font-semibold mb-4">Upload New Document</h2>
        <FileUpload onUploadSuccess={fetchDocuments} />
      </div>

      {/* Documents List */}
      <div>
        <h2 className="text-xl font-semibold mb-4">Your Documents</h2>

        {loading ? (
          <p className="text-gray-500">Loading documents...</p>
        ) : error ? (
          <p className="text-red-600">{error}</p>
        ) : documents.length === 0 ? (
          <p className="text-gray-500">No documents uploaded yet</p>
        ) : (
          <div className="grid gap-4">
            {documents.map((doc) => (
              <div
                key={doc._id}
                className="border rounded-lg p-4 flex justify-between items-center"
              >
                <div>
                  <p className="font-semibold">{doc.filename}</p>
                  <p className="text-sm text-gray-500">
                    {(doc.file_size_bytes / 1024 / 1024).toFixed(2)} MB •{' '}
                    {doc.chunks.count} chunks •{' '}
                    <span
                      className={`${
                        doc.status === 'processed'
                          ? 'text-green-600'
                          : doc.status === 'processing'
                          ? 'text-blue-600'
                          : 'text-red-600'
                      }`}
                    >
                      {doc.status}
                    </span>
                  </p>
                </div>
                <p className="text-sm text-gray-400">
                  {new Date(doc.created_at).toLocaleDateString()}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}