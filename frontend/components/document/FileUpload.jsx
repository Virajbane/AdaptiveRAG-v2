// frontend/components/document/FileUpload.jsx
'use client';

import { useState } from 'react';
import { useAuth } from '@/app/context/AuthContext';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// This component shows a drag-and-drop box for uploading files.
// It calls onUploadSuccess() when the upload finishes, so the parent
// page knows to refresh its document list.
//
// MULTI-FILE FIX (2026-07-14): previously both handleFileSelect and
// handleDrop only ever read files[0], discarding every other selected
// file -- and the <input> lacked the `multiple` attribute, so the OS
// picker dialog wouldn't even let you select more than one to begin
// with. The backend's /upload endpoint is intentionally single-file
// per request (one doc_id, one background task, one status-poll
// target per document) -- so the fix here is to keep that endpoint
// as-is and just fire one request per selected file from the
// frontend, tracked as a batch.
export function FileUpload({ onUploadSuccess }) {
  const { token } = useAuth();

  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');
  const [stage, setStage] = useState('idle'); // 'idle' | 'uploading' | 'queued'
  const [progress, setProgress] = useState({ done: 0, total: 0 });

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  // Uploads a single file. Returns { ok: true } or { ok: false, filename, message }
  // rather than throwing, so one failed file in a batch doesn't abort
  // the rest of the batch.
  const uploadOne = async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_URL}/api/v1/documents/upload`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        return { ok: false, filename: file.name, message: data.detail || 'Upload failed' };
      }

      await response.json();
      return { ok: true, filename: file.name };
    } catch (err) {
      return { ok: false, filename: file.name, message: err.message || 'Upload failed' };
    }
  };

  const uploadFiles = async (fileList) => {
    const files = Array.from(fileList);
    if (files.length === 0) return;

    setError('');
    setIsUploading(true);
    setStage('uploading');
    setProgress({ done: 0, total: files.length });

    // Fire uploads in parallel -- each gets its own doc_id and
    // background task server-side, so there's no reason to serialize
    // the HTTP requests themselves (only the same-filename-collision
    // edge case below is a caveat, not a reason to serialize).
    const results = await Promise.all(
      files.map(async (file) => {
        const result = await uploadOne(file);
        setProgress((prev) => ({ ...prev, done: prev.done + 1 }));
        return result;
      })
    );

    const failures = results.filter((r) => !r.ok);
    if (failures.length > 0) {
      const summary = failures
        .map((f) => `${f.filename}: ${f.message}`)
        .join('; ');
      setError(
        failures.length === files.length
          ? `All uploads failed — ${summary}`
          : `${failures.length} of ${files.length} uploads failed — ${summary}`
      );
    }

    setStage('queued');

    // Let the user see the "queued" state briefly, then hand control
    // back to the documents list -- which now polls on its own and
    // will show "processing" -> "processed"/"processed_with_gaps"/
    // "failed" as those states actually happen, live, for every file
    // in this batch.
    setTimeout(() => {
      setIsUploading(false);
      setStage('idle');
      setProgress({ done: 0, total: 0 });
      onUploadSuccess();
    }, 1200);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      uploadFiles(files);
    }
  };

  const handleFileSelect = (e) => {
    const files = e.currentTarget.files;
    if (files && files.length > 0) {
      uploadFiles(files);
    }
    // Reset so selecting the SAME file(s) again later still fires
    // onChange -- browsers don't fire change if the value is identical
    // to last time.
    e.currentTarget.value = '';
  };

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`border-2 border-dashed rounded-lg p-8 text-center transition ${
        isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-gray-50'
      } ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}
    >
      <input
        type="file"
        id="file-input"
        onChange={handleFileSelect}
        accept=".pdf,.docx,.txt,.csv"
        multiple
        className="hidden"
        disabled={isUploading}
      />

      {!isUploading ? (
        <label htmlFor="file-input" className="cursor-pointer">
          <div className="text-4xl mb-2">📄</div>
          <p className="text-lg font-semibold text-gray-900">
            Drag and drop your files here
          </p>
          <p className="text-sm text-gray-500 mt-2">
            or click to select (PDF, DOCX, TXT, CSV up to 50MB each — multiple files allowed)
          </p>
        </label>
      ) : (
        <div>
          <p className="text-lg font-semibold mb-2">
            {stage === 'uploading'
              ? progress.total > 1
                ? `Uploading files… (${progress.done}/${progress.total})`
                : 'Uploading file…'
              : progress.total > 1
              ? `${progress.total} files uploaded — processing started`
              : 'Uploaded — processing started'}
          </p>
          <p className="text-sm text-gray-500">
            {stage === 'uploading'
              ? 'Sending your file(s) to the server.'
              : "Chunking and embedding now. You'll see live status below once this closes."}
          </p>
          {/* Indeterminate bar, not a fake percentage -- there's no
              meaningful "% done" to show at the upload step, since
              real progress (chunking/embedding) hasn't started yet
              and happens server-side, tracked by the documents list's
              polling instead. */}
          <div className="w-full bg-gray-200 rounded-full h-2 mt-4 overflow-hidden">
            <div className="h-2 rounded-full bg-blue-600 animate-pulse w-full" />
          </div>
        </div>
      )}

      {error && <p className="text-red-600 text-sm mt-4 whitespace-pre-wrap">{error}</p>}
    </div>
  );
}