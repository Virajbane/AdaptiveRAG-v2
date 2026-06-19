'use client';

export function SearchResults({ results, loading }) {
  if (loading) {
    return <p className="text-gray-500">Searching...</p>;
  }

  if (!results || results.length === 0) {
    return <p className="text-gray-500">No results found</p>;
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600">{results.length} results found</p>
      
      {results.map((result, idx) => (
        <div
          key={idx}
          className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition"
        >
          {/* Score Indicators */}
          <div className="flex gap-4 mb-2 text-sm">
            <span className="text-blue-600">
              Relevance: {(result.combined_score * 100).toFixed(0)}%
            </span>
            {result.vector_score > 0 && (
              <span className="text-purple-600">
                Vector: {(result.vector_score * 100).toFixed(0)}%
              </span>
            )}
            {result.keyword_score > 0 && (
              <span className="text-green-600">
                Keyword: {(result.keyword_score * 100).toFixed(0)}%
              </span>
            )}
          </div>

          {/* Content */}
          <p className="text-gray-900 text-sm leading-relaxed">
            {result.text}
          </p>

          {/* Metadata */}
          <div className="mt-3 pt-3 border-t border-gray-100 text-xs text-gray-500">
            <span>Doc ID: {result.doc_id.substring(0, 8)}...</span>
            <span className="ml-4">Chunk {result.chunk_index}</span>
          </div>
        </div>
      ))}
    </div>
  );
}