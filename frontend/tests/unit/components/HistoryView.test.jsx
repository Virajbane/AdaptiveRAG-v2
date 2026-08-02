// tests/unit/components/HistoryView.test.jsx
//
// ASSUMPTION: source path is `components/layout/dashboard/views/HistoryView.jsx`,
// mirroring LibraryView.jsx / ChatView.jsx / SearchView.jsx's confirmed real
// location. Adjust the import below if wrong.
import HistoryView from '../../../components/layout/dashboard/views/HistoryView';

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

const API_URL = 'http://localhost:8000';
const TOKEN = 'test-token';

function jsonResponse(body, ok = true, status = ok ? 200 : 400) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) });
}

function makeMsg(overrides = {}) {
  return { role: 'user', content: 'What is the refund policy?', ...overrides };
}

function makeSummary(overrides = {}) {
  return { summary: 'Discussed refund and shipping policies.', topics: ['refunds', 'shipping'], ...overrides };
}

beforeEach(() => {
  vi.restoreAllMocks();
  window.confirm = vi.fn(() => true);
});

afterEach(() => {
  cleanup();
});

// ---- initial load -----------------------------------------------------------

describe('HistoryView — initial load', () => {
  it('loads memory on mount for the default session when a token is present', async () => {
    global.fetch = vi.fn(() => jsonResponse({ history: [], summaries: [] }));
    render(<HistoryView token={TOKEN} />);

    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));
    const [url, opts] = global.fetch.mock.calls[0];
    expect(url).toBe(`${API_URL}/api/v1/memory/load`);
    expect(opts.method).toBe('POST');
    expect(opts.headers).toEqual({ Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' });
    expect(JSON.parse(opts.body)).toEqual({ session_id: 'default_session' });
  });

  it('does not load memory on mount when there is no token', () => {
    global.fetch = vi.fn();
    render(<HistoryView token={null} />);
    expect(global.fetch).not.toHaveBeenCalled();
  });

  it('shows "Loading memory…" while the request is in flight, then hides it', async () => {
    let resolveFetch;
    global.fetch = vi.fn(() => new Promise((resolve) => { resolveFetch = resolve; }));
    render(<HistoryView token={TOKEN} />);

    expect(screen.getByText('Loading memory…')).toBeInTheDocument();
    resolveFetch({ ok: true, json: () => Promise.resolve({ history: [], summaries: [] }) });
    await waitFor(() => expect(screen.queryByText('Loading memory…')).not.toBeInTheDocument());
  });

  it('shows "No conversation history yet" once loaded with no messages', async () => {
    global.fetch = vi.fn(() => jsonResponse({ history: [], summaries: [] }));
    render(<HistoryView token={TOKEN} />);
    expect(await screen.findByText('No conversation history yet')).toBeInTheDocument();
  });

  it('shows an error message when the initial load fails', async () => {
    global.fetch = vi.fn(() => jsonResponse({}, false, 500));
    render(<HistoryView token={TOKEN} />);
    expect(await screen.findByText('Failed to load memory')).toBeInTheDocument();
  });
});

// ---- rendering messages and summaries ----------------------------------------

describe('HistoryView — rendering history and summaries', () => {
  it('renders the message count header and each message, with role and content', async () => {
    global.fetch = vi.fn(() => jsonResponse({
      history: [
        makeMsg({ role: 'user', content: 'What is the refund policy?' }),
        makeMsg({ role: 'assistant', content: 'Refunds are available within 30 days.' }),
      ],
      summaries: [],
    }));
    render(<HistoryView token={TOKEN} />);

    expect(await screen.findByText('Conversation History (2 messages)')).toBeInTheDocument();
    expect(screen.getByText('What is the refund policy?')).toBeInTheDocument();
    expect(screen.getByText('Refunds are available within 30 days.')).toBeInTheDocument();
  });

  it('shows a formatted timestamp only for messages that have one', async () => {
    global.fetch = vi.fn(() => jsonResponse({
      history: [
        makeMsg({ content: 'no timestamp on this one' }),
        makeMsg({ content: 'this one has a timestamp', timestamp: '2026-01-15T10:00:00Z' }),
      ],
      summaries: [],
    }));
    render(<HistoryView token={TOKEN} />);

    await screen.findByText('no timestamp on this one');
    // exactly one of the two messages should render a timestamp span
    const expected = new Date('2026-01-15T10:00:00Z').toLocaleString();
    expect(screen.getByText(expected)).toBeInTheDocument();
  });

  it('does not render the history section at all when there are no messages', async () => {
    global.fetch = vi.fn(() => jsonResponse({ history: [], summaries: [] }));
    render(<HistoryView token={TOKEN} />);
    await screen.findByText('No conversation history yet');
    expect(screen.queryByText(/Conversation History/)).not.toBeInTheDocument();
  });

  it('renders session summaries with their topic tags, only when present', async () => {
    global.fetch = vi.fn(() => jsonResponse({
      history: [],
      summaries: [makeSummary({ summary: 'Discussed refund and shipping.', topics: ['refunds', 'shipping'] })],
    }));
    render(<HistoryView token={TOKEN} />);

    expect(await screen.findByText('Session Summaries')).toBeInTheDocument();
    expect(screen.getByText('Discussed refund and shipping.')).toBeInTheDocument();
    expect(screen.getByText('refunds')).toBeInTheDocument();
    expect(screen.getByText('shipping')).toBeInTheDocument();
  });

  it('omits the topic tag row for a summary that has no topics', async () => {
    global.fetch = vi.fn(() => jsonResponse({
      history: [],
      summaries: [{ summary: 'A summary with no topics array.' }],
    }));
    render(<HistoryView token={TOKEN} />);
    expect(await screen.findByText('A summary with no topics array.')).toBeInTheDocument();
  });

  it('omits the summaries section entirely when there are no summaries', async () => {
    global.fetch = vi.fn(() => jsonResponse({ history: [makeMsg()], summaries: [] }));
    render(<HistoryView token={TOKEN} />);
    await screen.findByText('What is the refund policy?');
    expect(screen.queryByText('Session Summaries')).not.toBeInTheDocument();
  });
});

// ---- session switching --------------------------------------------------------

describe('HistoryView — session ID + Load button', () => {
  it('reloads memory for a new session id when the input is changed and Load is clicked', async () => {
    global.fetch = vi.fn(() => jsonResponse({ history: [], summaries: [] }));
    render(<HistoryView token={TOKEN} />);
    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(1));

    const input = screen.getByDisplayValue('default_session');
    const user = userEvent.setup();
    await user.clear(input);
    await user.type(input, 'session_42');
    await user.click(screen.getByRole('button', { name: 'Load' }));

    await waitFor(() => expect(global.fetch).toHaveBeenCalledTimes(2));
    const [, opts] = global.fetch.mock.calls[1];
    expect(JSON.parse(opts.body)).toEqual({ session_id: 'session_42' });
  });
});

// ---- clear session --------------------------------------------------------------

describe('HistoryView — clear session', () => {
  it('confirms before clearing, then deletes the session and empties messages/summaries', async () => {
    global.fetch = vi.fn((url, opts) => {
      if (opts?.method === 'DELETE') return jsonResponse({});
      return jsonResponse({
        history: [makeMsg()],
        summaries: [makeSummary()],
      });
    });
    render(<HistoryView token={TOKEN} />);
    await screen.findByText('What is the refund policy?');

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Clear' }));

    expect(window.confirm).toHaveBeenCalledWith('Clear this conversation?');
    await waitFor(() => expect(screen.queryByText('What is the refund policy?')).not.toBeInTheDocument());
    expect(screen.getByText('No conversation history yet')).toBeInTheDocument();
    expect(screen.queryByText('Session Summaries')).not.toBeInTheDocument();

    const deleteCall = global.fetch.mock.calls.find(([, opts]) => opts?.method === 'DELETE');
    expect(deleteCall[0]).toBe(`${API_URL}/api/v1/memory/session/default_session`);
  });

  it('does not clear anything when the confirm dialog is dismissed', async () => {
    window.confirm = vi.fn(() => false);
    global.fetch = vi.fn(() => jsonResponse({ history: [makeMsg()], summaries: [] }));
    render(<HistoryView token={TOKEN} />);
    await screen.findByText('What is the refund policy?');

    const fetchCallsBefore = global.fetch.mock.calls.length;
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Clear' }));

    expect(global.fetch.mock.calls.length).toBe(fetchCallsBefore);
    expect(screen.getByText('What is the refund policy?')).toBeInTheDocument();
  });

  // REAL BUG FOUND (more serious than initially assumed): clearSession
  // never checks `res.ok` before proceeding — it only has a try/catch
  // around the fetch call itself. fetch() resolves (doesn't throw) for
  // HTTP error responses like a 500; it only rejects on actual network
  // failures. So when the backend returns a 500, the code falls straight
  // through to setMessages([])/setSummaries([]) as if the delete
  // succeeded — silently wiping the visible conversation from the UI
  // even though nothing was actually deleted server-side, with zero
  // feedback that anything went wrong.
  it('GAP: an HTTP failure on delete is silently treated as success — messages get cleared with no error shown', async () => {
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    global.fetch = vi.fn((url, opts) => {
      if (opts?.method === 'DELETE') return jsonResponse({}, false, 500);
      return jsonResponse({ history: [makeMsg()], summaries: [] });
    });
    render(<HistoryView token={TOKEN} />);
    await screen.findByText('What is the refund policy?');

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Clear' }));

    await waitFor(() => expect(screen.queryByText('What is the refund policy?')).not.toBeInTheDocument());
    expect(screen.getByText('No conversation history yet')).toBeInTheDocument();
    expect(consoleErrorSpy).not.toHaveBeenCalled();
  });

  // Contrast case: a genuine network exception (fetch rejecting, not just
  // an HTTP error status) DOES hit the catch block and DOES get logged —
  // but there's still no user-visible error, and since the try block never
  // reached setMessages/setSummaries, the conversation stays on screen
  // rather than being wiped like in the 500 case above.
  it('logs a real network failure to console but still shows no visible error, and leaves messages untouched', async () => {
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    global.fetch = vi.fn((url, opts) => {
      if (opts?.method === 'DELETE') return Promise.reject(new Error('Network error'));
      return jsonResponse({ history: [makeMsg()], summaries: [] });
    });
    render(<HistoryView token={TOKEN} />);
    await screen.findByText('What is the refund policy?');

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Clear' }));

    await waitFor(() => expect(consoleErrorSpy).toHaveBeenCalled());
    expect(screen.getByText('What is the refund policy?')).toBeInTheDocument();
    expect(screen.queryByText('Failed to load memory')).not.toBeInTheDocument();
  });
});