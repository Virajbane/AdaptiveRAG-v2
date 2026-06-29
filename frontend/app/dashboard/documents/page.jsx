'use client';

import { useEffect, useState } from 'react';
import { useAuth } from '@/app/context/AuthContext';
import { FileUpload } from '@/components/document/FileUpload';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function DocumentsPage() {
  const { token } = useAuth();
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [deletingId, setDeletingId] = useState(null);

  const fetchDocuments = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/v1/documents`, {
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

  useEffect(() => {
    fetchDocuments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const handleDelete = async (docId) => {
    if (!window.confirm('Delete this document? This cannot be undone.')) return;
    setDeletingId(docId);
    try {
      const res = await fetch(`${API_URL}/api/v1/documents/${docId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) throw new Error('Failed to delete document');
      setDocuments((prev) => prev.filter((d) => d._id !== docId));
    } catch (err) {
      console.error('Error deleting document:', err);
      setError('Failed to delete document');
    } finally {
      setDeletingId(null);
    }
  };

  const statusColor = (status) => {
    if (status === 'processed') return '#34D399';
    if (status === 'processing') return '#60A5FA';
    return '#F87171';
  };

  return (
    <div className="mx-auto max-w-5xl p-6 lg:p-8">
      <div className="mb-8">
        <h1
          className="text-2xl font-medium tracking-tight text-white"
          style={{ fontFamily: 'Inter, sans-serif' }}
        >
          Documents
        </h1>
        <p className="mt-1 text-[14px]" style={{ color: '#A1A1AA' }}>
          Upload PDFs for your assistant to search and cite.
        </p>
      </div>

      {/* Upload */}
      <div className="mb-10 rounded-[11px] border border-white/10 bg-white/4 p-6 backdrop-blur-xl">
        <h2
          className="mb-4 text-[11px] font-semibold uppercase tracking-[0.08em]"
          style={{ fontFamily: 'JetBrains Mono, monospace', color: '#71717A' }}
        >
          Upload new document
        </h2>
        <FileUpload onUploadSuccess={fetchDocuments} />
      </div>

      {/* Documents list */}
      <div>
        <div className="mb-4 flex items-center justify-between">
          <h2
            className="text-[11px] font-semibold uppercase tracking-[0.08em]"
            style={{ fontFamily: 'JetBrains Mono, monospace', color: '#71717A' }}
          >
            Your documents
          </h2>
          {!loading && documents.length > 0 && (
            <span className="text-[13px]" style={{ color: '#71717A' }}>
              {documents.length} {documents.length === 1 ? 'document' : 'documents'}
            </span>
          )}
        </div>

        {loading ? (
          <p className="text-[14px]" style={{ color: '#A1A1AA' }}>Loading documents…</p>
        ) : error ? (
          <p className="text-[14px]" style={{ color: '#FCA5A5' }}>{error}</p>
        ) : documents.length === 0 ? (
          <div className="rounded-[11px] border border-white/10 bg-white/4 py-12 text-center backdrop-blur-xl">
            <p className="text-[14px]" style={{ color: '#A1A1AA' }}>No documents uploaded yet</p>
          </div>
        ) : (
          <div className="grid gap-3">
            {documents.map((doc) => (
              <div
                key={doc._id}
                className="flex items-center justify-between rounded-[11px] border border-white/10 bg-white/4 p-4 backdrop-blur-xl transition-colors hover:bg-white/6"
              >
                <div className="min-w-0">
                  <p className="truncate font-medium text-white">{doc.filename}</p>
                  <p className="text-[13px]" style={{ color: '#A1A1AA' }}>
                    {(doc.file_size_bytes / 1024 / 1024).toFixed(2)} MB •{' '}
                    {doc.chunks?.count ?? 0} chunks •{' '}
                    <span style={{ color: statusColor(doc.status) }}>{doc.status}</span>
                  </p>
                </div>
                <div className="ml-4 flex shrink-0 items-center gap-3">
                  <p className="text-[13px]" style={{ color: '#71717A' }}>
                    {new Date(doc.created_at).toLocaleDateString()}
                  </p>
                  <button
                    onClick={() => handleDelete(doc._id)}
                    disabled={deletingId === doc._id}
                    className="rounded-[8px] border px-2.5 py-1.5 text-[12px] font-medium transition-colors hover:bg-red-500/10 disabled:opacity-50"
                    style={{ borderColor: 'rgba(248,113,113,0.3)', color: '#FCA5A5' }}
                    aria-label={`Delete ${doc.filename}`}
                  >
                    {deletingId === doc._id ? 'Deleting…' : 'Delete'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}