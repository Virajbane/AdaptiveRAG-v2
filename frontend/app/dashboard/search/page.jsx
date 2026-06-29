'use client';

import { useState } from 'react';
import { SearchBox } from '@/components/chat/SearchBox';
import { SearchResults } from '@/components/chat/SearchResults';

export default function SearchPage() {
  const [results, setResults] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSearch = (searchResults) => {
    setResults(searchResults);
    setLoading(false);
  };

  return (
    <div className="mx-auto max-w-4xl p-6">
      <h1
        className="mb-6 text-2xl font-medium tracking-tight text-white"
        style={{ fontFamily: 'Inter, sans-serif' }}
      >
        Search Documents
      </h1>

      {/* Search Box */}
      <div className="mb-8 rounded-[11px] border border-white/10 bg-white/4 p-6 backdrop-blur-xl">
        <SearchBox onResults={handleSearch} />
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div>
          <h2
            className="mb-4 text-[11px] font-semibold uppercase tracking-[0.08em]"
            style={{ fontFamily: 'JetBrains Mono, monospace', color: '#71717A' }}
          >
            Results
          </h2>
          <SearchResults results={results} loading={loading} />
        </div>
      )}

      {/* Empty State */}
      {results.length === 0 && !loading && (
        <div className="rounded-[11px] border border-white/10 bg-white/4 py-12 text-center backdrop-blur-xl">
          <p className="text-[15px]" style={{ color: '#A1A1AA' }}>
            Start searching to find relevant documents
          </p>
        </div>
      )}
    </div>
  );
}