// frontend/components/common/Loading.jsx
'use client';

// A small reusable spinner. Pass a label to explain what's loading
// (e.g. "Loading documents..."), or leave it blank for a bare spinner.
export function Loading({ label = 'Loading...' }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-12 text-gray-500">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-gray-300 border-t-blue-600" />
      {label && <p className="text-sm">{label}</p>}
    </div>
  );
}