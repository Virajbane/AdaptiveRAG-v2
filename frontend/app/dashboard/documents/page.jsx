'use client';

import { useEffect, useState, useRef } from 'react';
import { useAuth } from '@/app/context/AuthContext';
import { FileUpload } from '@/components/document/FileUpload';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Statuses that mean "still working, keep polling"
const IN_PROGRESS_STATUSES = new Set(['processing']);

export default function DocumentsPage() {
  const { token } = useAuth();
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [deletingId, setDeletingId] = useState(null);
  const [retryingId, setRetryingId] = useState(null);
  const pollRef = useRef(null);

  const fetchDocuments = async ({ silent = false } = {}) => {
    if (!silent) setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/v1/documents`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) throw new Error('Failed to fetch documents');
      const data = await response.json();
      setDocuments(data.documents || []);
      return data.documents || [];
    } catch (err) {
      setError(err.message || 'Error loading documents');
      return [];
    } finally {
      if (!silent) setLoading(false);
    }
  };

  // Poll every 3s ONLY while at least one document is still "processing".
  // Stops itself once nothing is in progress, restarts automatically
  // next time a new upload flips a doc back to "processing".
  useEffect(() => {
    fetchDocuments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  useEffect(() => {
    const anyInProgress = documents.some((d) => IN_PROGRESS_STATUSES.has(d.status));

    if (anyInProgress && !pollRef.current) {
      pollRef.current = setInterval(() => {
        fetchDocuments({ silent: true }); // silent: don't flash the loading state every 3s
      }, 3000);
    }

    if (!anyInProgress && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documents]);

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

  const handleRetry = async (docId) => {
    setRetryingId(docId);
    try {
      const res = await fetch(`${API_URL}/api/v1/documents/${docId}/retry`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || 'Retry failed');
      }
      // Flip that doc to "processing" immediately in local state so the
      // UI updates right away, without waiting for the next poll tick.
      setDocuments((prev) =>
        prev.map((d) => (d._id === docId ? { ...d, status: 'processing' } : d))
      );
    } catch (err) {
      setError(err.message || 'Failed to retry document');
    } finally {
      setRetryingId(null);
    }
  };

  const statusColor = (status) => {
    if (status === 'processed') return '#34D399';
    if (status === 'processing') return '#60A5FA';
    if (status === 'processed_with_gaps') return '#FBBF24'; // amber — distinct from red (failed) and green (clean)
    return '#F87171'; // failed
  };

  const statusLabel = (status) => {
    if (status === 'processed_with_gaps') return 'processed (with gaps)';
    return status;
  };

  const canRetry = (status) => ['failed', 'processed_with_gaps', 'processing'].includes(status);

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
        <FileUpload onUploadSuccess={() => fetchDocuments()} />
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
            {documents.map((doc) => {
              const hasGapDetails =
                doc.status === 'processed_with_gaps' &&
                ((doc.chunks_failed ?? 0) > 0 || (doc.docling_page_errors?.length ?? 0) > 0);

              return (
                <div
                  key={doc._id}
                  className="rounded-[11px] border border-white/10 bg-white/4 p-4 backdrop-blur-xl transition-colors hover:bg-white/6"
                >
                  <div className="flex items-center justify-between">
                    <div className="min-w-0">
                      <p className="truncate font-medium text-white">{doc.filename}</p>
                      <p className="text-[13px]" style={{ color: '#A1A1AA' }}>
                        {(doc.file_size_bytes / 1024 / 1024).toFixed(2)} MB •{' '}
                        {doc.chunks?.count ?? 0} chunks •{' '}
                        <span style={{ color: statusColor(doc.status) }}>
                          {statusLabel(doc.status)}
                          {doc.status === 'processing' && ' …'}
                        </span>
                      </p>
                    </div>
                    <div className="ml-4 flex shrink-0 items-center gap-3">
                      <p className="text-[13px]" style={{ color: '#71717A' }}>
                        {new Date(doc.created_at).toLocaleDateString()}
                      </p>
                      {canRetry(doc.status) && doc.status !== 'processing' && (
                        <button
                          onClick={() => handleRetry(doc._id)}
                          disabled={retryingId === doc._id}
                          className="rounded-[8px] border px-2.5 py-1.5 text-[12px] font-medium transition-colors hover:bg-blue-500/10 disabled:opacity-50"
                          style={{ borderColor: 'rgba(96,165,250,0.3)', color: '#93C5FD' }}
                        >
                          {retryingId === doc._id ? 'Retrying…' : 'Retry'}
                        </button>
                      )}
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

                  {/* Gap details -- only shown for processed_with_gaps, only
                      the specific gap types that actually occurred */}
                  {hasGapDetails && (
                    <div
                      className="mt-3 rounded-[8px] border px-3 py-2 text-[12px]"
                      style={{ borderColor: 'rgba(251,191,36,0.25)', color: '#FCD34D', background: 'rgba(251,191,36,0.06)' }}
                    >
                      <p className="font-medium mb-1">⚠️ This document indexed with gaps:</p>
                      {(doc.chunks_failed ?? 0) > 0 && (
                        <p>• {doc.chunks_failed} chunk(s) failed to store — retry to attempt again.</p>
                      )}
                      {(doc.docling_page_errors?.length ?? 0) > 0 && (
                        <p>
                          • {doc.docling_page_errors.length} page(s) failed during parsing —
                          this usually needs a full reprocess.
                        </p>
                      )}
                    </div>
                  )}

                  {doc.status === 'failed' && doc.processing_error && (
                    <div
                      className="mt-3 rounded-[8px] border px-3 py-2 text-[12px]"
                      style={{ borderColor: 'rgba(248,113,113,0.25)', color: '#FCA5A5', background: 'rgba(248,113,113,0.06)' }}
                    >
                      <p className="font-medium">❌ Processing failed: {doc.processing_error}</p>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}