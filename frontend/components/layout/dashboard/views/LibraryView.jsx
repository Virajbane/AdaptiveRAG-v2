'use client';

// Library view: upload dropzone + documents list with delete + retry.
// All document-specific state and API logic lives here — if uploads,
// deletes, or retries break, this is the only file you need to touch/paste.
//
// MULTI-FILE FIX (2026-07-14): previously the <input> had no `multiple`
// attribute (so the OS picker dialog itself wouldn't allow selecting more
// than one file), and both onDrop and onChange hardcoded files[0],
// discarding every other file even when multiple were dropped at once.
// handleUpload now accepts a FileList/array and uploads every file in it
// in parallel (each gets its own doc_id/background task server-side,
// same as the single-file endpoint always assumed), then refreshes the
// document list ONCE at the end instead of once per file.
//
// RETRY BUTTON ADDED: the backend has had POST /documents/{doc_id}/retry
// since before this file was last touched, but nothing in the UI ever
// called it -- a "failed" or "processed_with_gaps" document had no way
// to be fixed short of deleting and re-uploading the whole file. Retry
// is only shown for those two statuses, matching the backend's own
// guard (retry_document rejects anything else with a 400), so this
// never renders a button that would just error when clicked.
//
// POLLING ADDED: FileUpload.jsx's own comments already assumed this view
// "polls on its own" after an upload -- it didn't. A document could
// finish processing entirely on the backend (see server logs: chunked,
// embedded, stored) while the screen kept showing "processing" until the
// user manually refreshed. Now, whenever any document is in "processing",
// this view quietly re-fetches every few seconds (no loading spinner) and
// stops on its own once nothing is in-flight anymore.

import { useState, useRef, useEffect } from 'react';
import { S } from '../styles';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Statuses the backend's /retry endpoint will actually accept.
// ("processing" is also technically retryable server-side, but only
// when stale -- there's no reliable way to tell that from this list
// payload alone, so we don't offer retry for "processing" here to
// avoid a button that 409s on a genuinely still-running job.)
const RETRYABLE_STATUSES = new Set(['failed', 'processed_with_gaps']);

// How often to re-check the list while something is still "processing".
// Docling parsing + embedding a real paper can take several minutes
// (see backend logs), so this doesn't need to be aggressive -- it just
// needs to exist, since right now nothing ever re-fetches after the
// initial load/upload, and a document that finishes processing in the
// background sits showing a stale "processing" badge until the user
// manually refreshes the page.
const POLL_INTERVAL_MS = 4000;

export default function LibraryView({ token, onDocumentsChange }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [deletingId, setDeletingId] = useState(null);
  const [retryingId, setRetryingId] = useState(null);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);

  // `silent` skips the loading spinner -- used by the background poll so
  // a routine "is it done yet?" check doesn't hide the whole list and
  // flash "Loading…" every few seconds. The real initial page-load fetch
  // still shows the spinner as before.
  const fetchDocuments = async ({ silent = false } = {}) => {
    if (!silent) setLoading(true);
    setError('');
    try {
      const res = await fetch(`${API_URL}/api/v1/documents`, { headers: { Authorization: `Bearer ${token}` } });
      if (!res.ok) throw new Error('Failed to fetch documents');
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (err) {
      // Don't surface a visible error for a silent background poll --
      // a single dropped poll tick will just retry next interval. Only
      // the user-initiated (non-silent) fetches should show an error.
      if (!silent) setError(err.message);
    } finally {
      if (!silent) setLoading(false);
    }
  };

  useEffect(() => {
    if (token) fetchDocuments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  // Let the parent (Dashboard) know the doc count/list — used for the
  // sidebar "Workspace Docs" shortcuts and the chat "Searching N files" text.
  useEffect(() => { onDocumentsChange?.(documents); }, [documents]); // eslint-disable-line react-hooks/exhaustive-deps

  // POLLING: while any document is still "processing", keep re-fetching
  // the list on an interval so the UI catches the status change on its
  // own instead of freezing at "processing" until the user manually
  // reloads the page. Stops itself the moment nothing is in-flight
  // anymore -- no wasted requests once everything has settled into
  // processed / processed_with_gaps / failed.
  useEffect(() => {
    const hasInFlight = documents.some(d => d.status === 'processing');
    if (!hasInFlight) return;

    const id = setInterval(() => {
      fetchDocuments({ silent: true });
    }, POLL_INTERVAL_MS);

    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documents]);

  // Uploads a single file. Returns { ok, filename, message? } instead of
  // throwing, so one bad file in a batch doesn't abort the rest.
  const uploadOne = async (file) => {
    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${API_URL}/api/v1/documents/upload`, { method: 'POST', headers: { Authorization: `Bearer ${token}` }, body: fd });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        return { ok: false, filename: file.name, message: data.detail || 'Upload failed' };
      }
      return { ok: true, filename: file.name };
    } catch (err) {
      return { ok: false, filename: file.name, message: err.message || 'Upload failed' };
    }
  };

  // Accepts a FileList or array of File objects -- one or many. Uploads
  // all of them in parallel, then refreshes the document list once.
  const handleUpload = async (fileListOrArray) => {
    const files = Array.from(fileListOrArray || []).filter(Boolean);
    if (files.length === 0) return;

    setUploading(true);
    setError('');
    try {
      const results = await Promise.all(files.map(uploadOne));
      const failures = results.filter(r => !r.ok);
      if (failures.length > 0) {
        const summary = failures.map(f => `${f.filename}: ${f.message}`).join('; ');
        setError(
          failures.length === files.length
            ? `All uploads failed — ${summary}`
            : `${failures.length} of ${files.length} uploads failed — ${summary}`
        );
      }
    } finally {
      setUploading(false);
      await fetchDocuments();
    }
  };

  const handleDelete = async (docId) => {
    if (!window.confirm('Delete this document? This cannot be undone.')) return;
    setDeletingId(docId);
    try {
      const res = await fetch(`${API_URL}/api/v1/documents/${docId}`, { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } });
      if (!res.ok) throw new Error('Delete failed');
      setDocuments(p => p.filter(d => d._id !== docId));
    } catch (err) {
      setError(err.message);
    } finally {
      setDeletingId(null);
    }
  };

  const handleRetry = async (docId) => {
    setRetryingId(docId);
    setError('');
    try {
      const res = await fetch(`${API_URL}/api/v1/documents/${docId}/retry`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || 'Retry failed');
      }
      // Backend re-queues in the background and returns status:"processing"
      // immediately -- reflect that right away so the row updates without
      // waiting on a poll cycle, then do a real fetch to pick up whatever
      // the poller/refresh cadence would show next anyway.
      setDocuments(prev => prev.map(d => (d._id === docId ? { ...d, status: 'processing' } : d)));
      await fetchDocuments();
    } catch (err) {
      setError(err.message);
    } finally {
      setRetryingId(null);
    }
  };

  const statusColor = (s) => s === 'processed' ? '#1ed760' : s === 'processing' ? '#539df5' : '#f3727f';

  return (
    <div style={{ padding: '32px 32px', overflowY: 'auto', height: 'calc(100vh - 64px)' }}>
      <div style={{ maxWidth: 860, margin: '0 auto' }}>
        <h1 style={S.pageTitle}>Knowledge Base</h1>
        <p style={S.pageSubtitle}>Upload PDFs for your assistant to search and cite.</p>

        {/* Upload zone */}
        <div
          style={{ ...S.card, padding: 32, marginBottom: 28, textAlign: 'center', cursor: 'pointer', border: '2px dashed rgba(255,255,255,0.1)', transition: 'border-color 0.2s' }}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); e.currentTarget.style.borderColor = '#1ed760'; }}
          onDragLeave={e => e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'}
          onDrop={e => { e.preventDefault(); e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'; if (e.dataTransfer.files.length > 0) handleUpload(e.dataTransfer.files); }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.docx"
            multiple
            style={{ display: 'none' }}
            onChange={e => {
              if (e.target.files && e.target.files.length > 0) handleUpload(e.target.files);
              e.target.value = ''; // allow re-selecting the same file(s) later
            }}
          />
          <div style={{ width: 48, height: 48, borderRadius: '50%', background: '#1f1f1f', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px' }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#1ed760" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17,8 12,3 7,8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
          </div>
          <p style={{ fontSize: 15, color: '#e5e2e1', marginBottom: 4 }}>{uploading ? 'Uploading…' : 'Drop files here or click to upload'}</p>
          <p style={{ fontSize: 12, color: '#b3b3b3' }}>Supports PDF, TXT, DOCX — multiple files allowed</p>
        </div>

        {/* Header row */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#b3b3b3' }}>Your documents</span>
          {!loading && <span style={{ fontSize: 13, color: '#4d4d4d' }}>{documents.length} {documents.length === 1 ? 'file' : 'files'}</span>}
        </div>

        {loading && <p style={{ color: '#b3b3b3', fontSize: 14 }}>Loading…</p>}
        {error && <p style={{ color: '#f3727f', fontSize: 14, whiteSpace: 'pre-wrap' }}>{error}</p>}

        {!loading && documents.length === 0 && (
          <div style={{ ...S.card, padding: 48, textAlign: 'center' }}>
            <p style={{ color: '#b3b3b3', fontSize: 14 }}>No documents uploaded yet</p>
          </div>
        )}

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {documents.map(doc => (
            <div key={doc._id} style={{ ...S.card, display: 'flex', alignItems: 'center', padding: '14px 16px', gap: 14 }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#1ed760" strokeWidth="1.8" style={{ flexShrink: 0 }}>
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M9 13h6M9 17h4"/>
              </svg>
              <div style={{ flex: 1, overflow: 'hidden' }}>
                <p style={{ fontSize: 14, fontWeight: 600, color: '#e5e2e1', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', margin: 0 }}>{doc.filename}</p>
                <p style={{ fontSize: 12, color: '#b3b3b3', margin: 0, marginTop: 2 }}>
                  {(doc.file_size_bytes / 1024 / 1024).toFixed(2)} MB • {doc.chunks?.count ?? 0} chunks •{' '}
                  <span style={{ color: statusColor(doc.status) }}>{doc.status}</span>
                  {doc.status === 'processed_with_gaps' && doc.docling_page_errors?.length > 0 && (
                    <span style={{ color: '#b3b3b3' }}> ({doc.docling_page_errors.length} page(s) had parse issues)</span>
                  )}
                </p>
              </div>
              <span style={{ fontSize: 12, color: '#4d4d4d', flexShrink: 0 }}>{new Date(doc.created_at).toLocaleDateString()}</span>

              {RETRYABLE_STATUSES.has(doc.status) && (
                <button
                  onClick={() => handleRetry(doc._id)}
                  disabled={retryingId === doc._id}
                  style={{ padding: '5px 12px', borderRadius: 9999, border: '1px solid rgba(83,157,245,0.3)', background: 'transparent', color: '#539df5', fontSize: 12, cursor: 'pointer', flexShrink: 0, opacity: retryingId === doc._id ? 0.5 : 1 }}
                >
                  {retryingId === doc._id ? 'Retrying…' : 'Retry'}
                </button>
              )}

              <button
                onClick={() => handleDelete(doc._id)}
                disabled={deletingId === doc._id}
                style={{ padding: '5px 12px', borderRadius: 9999, border: '1px solid rgba(243,114,127,0.3)', background: 'transparent', color: '#f3727f', fontSize: 12, cursor: 'pointer', flexShrink: 0, opacity: deletingId === doc._id ? 0.5 : 1 }}
              >
                {deletingId === doc._id ? 'Deleting…' : 'Delete'}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}