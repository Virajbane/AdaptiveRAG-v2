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
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Search Documents</h1>

      {/* Search Box */}
      <div className="mb-8">
        <SearchBox onResults={handleSearch} />
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div>
          <h2 className="text-xl font-semibold mb-4">Results</h2>
          <SearchResults results={results} loading={loading} />
        </div>
      )}

      {/* Empty State */}
      {results.length === 0 && !loading && (
        <div className="text-center py-12 text-gray-500">
          <p className="text-lg">Start searching to find relevant documents</p>
        </div>
      )}
    </div>
  );
}