// frontend/components/common/ErrorBoundary.jsx
'use client';

import { Component } from 'react';

// React error boundaries must be class components - there's no hook
// equivalent yet. This catches any rendering crash in its children
// and shows a friendly message instead of a blank white screen.
export class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    // Log to console for now - could send to a logging service later
    console.error('ErrorBoundary caught:', error, info);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
          <p className="text-lg font-medium text-gray-900">Something went wrong</p>
          <p className="text-sm text-gray-500 max-w-sm">
            {this.state.error?.message || 'An unexpected error occurred.'}
          </p>
          <button
            onClick={this.handleReset}
            className="mt-2 px-4 py-2 text-sm font-medium text-white bg-gray-900 rounded-md hover:bg-gray-700"
          >
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}