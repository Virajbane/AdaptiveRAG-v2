// tests/unit/components/SearchView.test.jsx
//
// ASSUMPTION: source path is `components/layout/dashboard/views/SearchView.jsx`,
// mirroring LibraryView.jsx and ChatView.jsx's confirmed real location.
// Adjust the import below if wrong.
import SearchView from '../../../components/layout/dashboard/views/SearchView';

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent, waitFor, act } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

const API_URL = 'http://localhost:8000';
const TOKEN = 'test-token';

function jsonResponse(body, ok = true, status = ok ? 200 : 400) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) });
}

function makeResult(overrides = {}) {
  return {
    filename: 'handbook.pdf',
    content: 'Employees are entitled to 15 days of paid leave per year.',
    score: 0.91,
    page: 4,
    ...overrides,
  };
}

beforeEach(() => {
  vi.restoreAllMocks();
});

afterEach(() => {
  cleanup();
});

// ---- initial / empty state -------------------------------------------------

describe('SearchView — initial state', () => {
  it('shows the "enter a query" placeholder and does not call fetch on mount', () => {
    global.fetch = vi.fn();
    render(<SearchView token={TOKEN} />);
    expect(screen.getByText('Enter a query to search your documents')).toBeInTheDocument();
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('disables the Search button when the query is empty or whitespace-only', async () => {
    render(<SearchView token={TOKEN} />);
    const input = screen.getByPlaceholderText('Search your documents…');
    const button = screen.getByRole('button', { name: /^Search$/ });

    expect(button).toBeDisabled();

    const user = userEvent.setup();
    await user.type(input, '   ');
    expect(button).toBeDisabled();

    await user.type(input, 'leave policy');
    expect(button).not.toBeDisabled();
  });
});

// ---- executing a search -----------------------------------------------------

describe('SearchView — executing a search', () => {
  it('calls the search endpoint with an encoded query, top_k=10, and the auth header', async () => {
    global.fetch = vi.fn(() => jsonResponse({ results: [] }));
    render(<SearchView token={TOKEN} />);

    const input = screen.getByPlaceholderText('Search your documents…');
    const user = userEvent.setup();
    await user.type(input, 'paid leave & holidays');
    await user.click(screen.getByRole('button', { name: /^Search$/ }));

    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    const [url, opts] = global.fetch.mock.calls[0];
    expect(url).toBe(`${API_URL}/api/v1/search?q=${encodeURIComponent('paid leave & holidays')}&top_k=10`);
    expect(opts.headers).toEqual({ Authorization: `Bearer ${TOKEN}` });
  });

  it('also searches when Enter is pressed in the input', async () => {
    global.fetch = vi.fn(() => jsonResponse({ results: [] }));
    render(<SearchView token={TOKEN} />);

    const input = screen.getByPlaceholderText('Search your documents…');
    const user = userEvent.setup();
    await user.type(input, 'onboarding steps');
    fireEvent.keyDown(input, { key: 'Enter' });

    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
  });

  it('does not search when Enter is pressed with an empty query', () => {
    global.fetch = vi.fn();
    render(<SearchView token={TOKEN} />);
    const input = screen.getByPlaceholderText('Search your documents…');
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('shows "Searching…" and disables the button while a search is in flight', async () => {
    let resolveFetch;
    global.fetch = vi.fn(() => new Promise((resolve) => { resolveFetch = resolve; }));
    render(<SearchView token={TOKEN} />);

    const input = screen.getByPlaceholderText('Search your documents…');
    const user = userEvent.setup();
    await user.type(input, 'benefits');
    await user.click(screen.getByRole('button', { name: /^Search$/ }));

    expect(screen.getByRole('button', { name: /^Searching…$/ })).toBeDisabled();

    resolveFetch({ ok: true, json: () => Promise.resolve({ results: [] }) });
    await waitFor(() => expect(screen.getByRole('button', { name: /^Search$/ })).toBeInTheDocument());
  });

  it('hides the "enter a query" empty-state message while a search is in flight', async () => {
    let resolveFetch;
    global.fetch = vi.fn(() => new Promise((resolve) => { resolveFetch = resolve; }));
    render(<SearchView token={TOKEN} />);

    const input = screen.getByPlaceholderText('Search your documents…');
    const user = userEvent.setup();
    await user.type(input, 'benefits');
    await user.click(screen.getByRole('button', { name: /^Search$/ }));

    expect(screen.queryByText('Enter a query to search your documents')).not.toBeInTheDocument();
    await act(async () => {
      resolveFetch({ ok: true, json: () => Promise.resolve({ results: [] }) });
      await Promise.resolve();
    });
  });
});

// ---- result rendering --------------------------------------------------------

describe('SearchView — result rendering', () => {
  it('renders filename, content, score percentage, and page for each result', async () => {
    global.fetch = vi.fn(() => jsonResponse({
      results: [makeResult({ filename: 'handbook.pdf', content: '15 days of paid leave.', score: 0.91, page: 4 })],
    }));
    render(<SearchView token={TOKEN} />);

    const input = screen.getByPlaceholderText('Search your documents…');
    const user = userEvent.setup();
    await user.type(input, 'leave');
    await user.click(screen.getByRole('button', { name: /^Search$/ }));

    expect(await screen.findByText('handbook.pdf')).toBeInTheDocument();
    expect(screen.getByText('15 days of paid leave.')).toBeInTheDocument();
    expect(screen.getByText('91% match')).toBeInTheDocument();
    expect(screen.getByText('Page 4')).toBeInTheDocument();
  });

  it('falls back to `source` for the title and `text` for the body when filename/content are absent', async () => {
    global.fetch = vi.fn(() => jsonResponse({
      results: [{ source: 'legacy-doc.txt', text: 'Older ingestion pipeline result.' }],
    }));
    render(<SearchView token={TOKEN} />);

    const input = screen.getByPlaceholderText('Search your documents…');
    const user = userEvent.setup();
    await user.type(input, 'legacy');
    await user.click(screen.getByRole('button', { name: /^Search$/ }));

    expect(await screen.findByText('legacy-doc.txt')).toBeInTheDocument();
    expect(screen.getByText('Older ingestion pipeline result.')).toBeInTheDocument();
  });

  it('falls back to "Result N" when neither filename nor source is present', async () => {
    global.fetch = vi.fn(() => jsonResponse({
      results: [{ content: 'no title here' }, { content: 'still no title' }],
    }));
    render(<SearchView token={TOKEN} />);

    const input = screen.getByPlaceholderText('Search your documents…');
    const user = userEvent.setup();
    await user.type(input, 'x');
    await user.click(screen.getByRole('button', { name: /^Search$/ }));

    expect(await screen.findByText('Result 1')).toBeInTheDocument();
    expect(screen.getByText('Result 2')).toBeInTheDocument();
  });

  it('omits the match percentage when score is undefined, and omits the page line when page is absent', async () => {
    global.fetch = vi.fn(() => jsonResponse({
      results: [{ filename: 'no-score.pdf', content: 'no score or page on this one' }],
    }));
    render(<SearchView token={TOKEN} />);

    const input = screen.getByPlaceholderText('Search your documents…');
    const user = userEvent.setup();
    await user.type(input, 'x');
    await user.click(screen.getByRole('button', { name: /^Search$/ }));

    await screen.findByText('no-score.pdf');
    expect(screen.queryByText(/% match/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Page /)).not.toBeInTheDocument();
  });

  it('clears previous results before showing new ones on a fresh search', async () => {
    global.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve({ results: [makeResult({ filename: 'first.pdf' })] }) })
      .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve({ results: [makeResult({ filename: 'second.pdf' })] }) });

    render(<SearchView token={TOKEN} />);
    const input = screen.getByPlaceholderText('Search your documents…');
    const user = userEvent.setup();

    await user.type(input, 'first query');
    await user.click(screen.getByRole('button', { name: /^Search$/ }));
    expect(await screen.findByText('first.pdf')).toBeInTheDocument();

    await user.clear(input);
    await user.type(input, 'second query');
    await user.click(screen.getByRole('button', { name: /^Search$/ }));

    expect(await screen.findByText('second.pdf')).toBeInTheDocument();
    expect(screen.queryByText('first.pdf')).not.toBeInTheDocument();
  });
});

// ---- error handling -----------------------------------------------------------

describe('SearchView — error handling', () => {
  it('shows an error message and clears results when the search request fails', async () => {
    global.fetch = vi.fn(() => jsonResponse({}, false, 500));
    render(<SearchView token={TOKEN} />);

    const input = screen.getByPlaceholderText('Search your documents…');
    const user = userEvent.setup();
    await user.type(input, 'anything');
    await user.click(screen.getByRole('button', { name: /^Search$/ }));

    expect(await screen.findByText('Search failed')).toBeInTheDocument();
  });

  it('clears a previous error on the next successful search', async () => {
    global.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: false, status: 500, json: () => Promise.resolve({}) })
      .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve({ results: [makeResult()] }) });

    render(<SearchView token={TOKEN} />);
    const input = screen.getByPlaceholderText('Search your documents…');
    const user = userEvent.setup();

    await user.type(input, 'first attempt');
    await user.click(screen.getByRole('button', { name: /^Search$/ }));
    expect(await screen.findByText('Search failed')).toBeInTheDocument();

    await user.clear(input);
    await user.type(input, 'second attempt');
    await user.click(screen.getByRole('button', { name: /^Search$/ }));

    await waitFor(() => expect(screen.queryByText('Search failed')).not.toBeInTheDocument());
  });
});