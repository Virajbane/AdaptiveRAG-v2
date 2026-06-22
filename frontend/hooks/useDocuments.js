// frontend/hooks/useDocuments.js
'use client';

import { useState, useEffect, useCallback } from 'react';
import { apiGet, apiUpload, apiDelete } from '@/lib/api';

// Manages the user's document list: fetching it, uploading a new
// file, and deleting an existing one. Re-fetches the list after
// each upload/delete so the UI always shows current state.
export function useDocuments() {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);

  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const data = await apiGet('/documents');
      setDocuments(data.documents || []);
    } catch (err) {
      setError(err.message || 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const uploadDocument = async (file) => {
    setUploading(true);
    setError('');
    try {
      await apiUpload('/documents/upload', file);
      await fetchDocuments(); // refresh the list to show the new doc
    } catch (err) {
      setError(err.message || 'Upload failed');
      throw err; // let the calling component know it failed too
    } finally {
      setUploading(false);
    }
  };

  const deleteDocument = async (docId) => {
    try {
      await apiDelete(`/documents/${docId}`);
      await fetchDocuments();
    } catch (err) {
      setError(err.message || 'Delete failed');
      throw err;
    }
  };

  return {
    documents,
    loading,
    uploading,
    error,
    uploadDocument,
    deleteDocument,
    refresh: fetchDocuments,
  };
}