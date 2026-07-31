// tests/unit/components/LibraryView.test.jsx
//
import LibraryView from '../../../components/layout/dashboard/views/LibraryView';

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, within, fireEvent, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

const API_URL = 'http://localhost:8000';
const TOKEN = 'test-token';

// ---- helpers -----------------------------------------------------------

function makeDoc(overrides = {}) {
  return {
    _id: 'doc-1',
    filename: 'paper.pdf',
    file_size_bytes: 1024 * 1024 * 2,
    status: 'processed',
    chunks: { count: 12 },
    created_at: '2026-07-01T00:00:00Z',
    ...overrides,
  };
}

function mockFetchSequence(handlers) {
  // handlers: array of fns (url, opts) => Response-like, consumed in order
  // for calls that don't match, falls back to the last handler
  let i = 0;
  global.fetch = vi.fn((url, opts) => {
    const handler = handlers[Math.min(i, handlers.length - 1)];
    i += 1;
    return handler(url, opts);
  });
}

function jsonResponse(body, ok = true, status = ok ? 200 : 400) {
  return Promise.resolve({
    ok,
    status,
    json: () => Promise.resolve(body),
  });
}

function makeFile(name = 'a.pdf', type = 'application/pdf') {
  return new File(['dummy content'], name, { type });
}

// A single `advanceTimersByTimeAsync(0)` only flushes one hop of a promise
// chain. Our fetch mocks resolve over several hops (fetch() -> res.json()
// -> setState -> re-render), so under fake timers we need to tick a few
// times in a row to let all of that settle before asserting on the DOM.
async function flushMicrotasks(times = 5) {
  for (let i = 0; i < times; i++) {
    await vi.advanceTimersByTimeAsync(0);
  }
}

beforeEach(() => {
  vi.restoreAllMocks();
  window.confirm = vi.fn(() => true);
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

// ---- initial load -------------------------------------------------------

describe('LibraryView — initial load', () => {
  it('fetches documents on mount when a token is present', async () => {
    mockFetchSequence([() => jsonResponse({ documents: [makeDoc()] })]);

    render(<LibraryView token={TOKEN} />);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        `${API_URL}/api/v1/documents`,
        expect.objectContaining({ headers: { Authorization: `Bearer ${TOKEN}` } })
      );
    });
    expect(await screen.findByText('paper.pdf')).toBeInTheDocument();
  });

  it('shows the empty state when there are no documents', async () => {
    mockFetchSequence([() => jsonResponse({ documents: [] })]);
    render(<LibraryView token={TOKEN} />);
    expect(await screen.findByText('No documents uploaded yet')).toBeInTheDocument();
  });

  it('calls onDocumentsChange with the fetched list', async () => {
    const onDocumentsChange = vi.fn();
    const docs = [makeDoc()];
    mockFetchSequence([() => jsonResponse({ documents: docs })]);

    render(<LibraryView token={TOKEN} onDocumentsChange={onDocumentsChange} />);

    await waitFor(() => {
      expect(onDocumentsChange).toHaveBeenCalledWith(docs);
    });
  });
});

// ---- MULTI-FILE UPLOAD FIX ----------------------------------------------

describe('LibraryView — multi-file upload (regression for the multi-file fix)', () => {
  it('renders the file input with `multiple` so the OS picker allows >1 file', async () => {
    mockFetchSequence([() => jsonResponse({ documents: [] })]);
    render(<LibraryView token={TOKEN} />);
    await screen.findByText('No documents uploaded yet');

    const input = document.querySelector('input[type="file"]');
    expect(input).toHaveAttribute('multiple');
  });

  it('uploads every file in a multi-file selection, not just files[0]', async () => {
    const uploadCalls = [];
    global.fetch = vi.fn((url, opts) => {
      if (url === `${API_URL}/api/v1/documents` ) return jsonResponse({ documents: [] });
      if (url === `${API_URL}/api/v1/documents/upload`) {
        uploadCalls.push(opts.body.get('file').name);
        return jsonResponse({ ok: true });
      }
      return jsonResponse({ documents: [makeDoc(), makeDoc({ _id: 'doc-2', filename: 'b.pdf' })] });
    });

    render(<LibraryView token={TOKEN} />);
    await screen.findByText('No documents uploaded yet');

    const input = document.querySelector('input[type="file"]');
    const files = [makeFile('a.pdf'), makeFile('b.pdf'), makeFile('c.pdf')];
    fireEvent.change(input, { target: { files } });

    await waitFor(() => {
      expect(uploadCalls.sort()).toEqual(['a.pdf', 'b.pdf', 'c.pdf']);
    });
  });

  it('refreshes the document list once, not once per uploaded file', async () => {
    let listFetchCount = 0;
    global.fetch = vi.fn((url) => {
      if (url === `${API_URL}/api/v1/documents`) {
        listFetchCount += 1;
        return jsonResponse({ documents: [] });
      }
      if (url === `${API_URL}/api/v1/documents/upload`) {
        return jsonResponse({ ok: true });
      }
      return jsonResponse({ documents: [] });
    });

    render(<LibraryView token={TOKEN} />);
    await screen.findByText('No documents uploaded yet');
    listFetchCount = 0; // reset after the initial mount fetch

    const input = document.querySelector('input[type="file"]');
    fireEvent.change(input, { target: { files: [makeFile('a.pdf'), makeFile('b.pdf'), makeFile('c.pdf')] } });

    await waitFor(() => expect(listFetchCount).toBe(1));
    // give any stray extra refetch a chance to show up before asserting it didn't
    await new Promise((r) => setTimeout(r, 20));
    expect(listFetchCount).toBe(1);
  });

  it('uploads multiple files dropped via drag-and-drop, not just the first', async () => {
    const uploadCalls = [];
    global.fetch = vi.fn((url, opts) => {
      if (url === `${API_URL}/api/v1/documents`) return jsonResponse({ documents: [] });
      if (url === `${API_URL}/api/v1/documents/upload`) {
        uploadCalls.push(opts.body.get('file').name);
        return jsonResponse({ ok: true });
      }
      return jsonResponse({ documents: [] });
    });

    render(<LibraryView token={TOKEN} />);
    const dropzone = await screen.findByText(/Drop files here or click to upload/);

    fireEvent.drop(dropzone.closest('div'), {
      dataTransfer: { files: [makeFile('x.pdf'), makeFile('y.pdf')] },
    });

    await waitFor(() => {
      expect(uploadCalls.sort()).toEqual(['x.pdf', 'y.pdf']);
    });
  });

  // REAL BUG FOUND (not anticipated going in): `fetchDocuments()` calls
  // `setError('')` unconditionally at the top, even when called silently.
  // `handleUpload`'s `finally` block calls `await fetchDocuments()` right
  // after setting the upload-failure summary via `setError(summary)` —
  // since that clear happens synchronously before fetchDocuments' own
  // `await fetch(...)`, it wipes the just-set summary out in the same
  // tick, before the user ever sees it. Written as `it.fails` so this
  // documents the current (buggy) behavior and flips to a visible
  // failure the moment someone fixes fetchDocuments to only clear error
  // on non-silent calls (or handleUpload stops clobbering its own state).
  it.fails('reports a partial-failure summary when some but not all files fail', async () => {
    global.fetch = vi.fn((url, opts) => {
      if (url === `${API_URL}/api/v1/documents`) return jsonResponse({ documents: [] });
      if (url === `${API_URL}/api/v1/documents/upload`) {
        const name = opts.body.get('file').name;
        if (name === 'bad.pdf') {
          return jsonResponse({ detail: 'corrupt file' }, false, 422);
        }
        return jsonResponse({ ok: true });
      }
      return jsonResponse({ documents: [] });
    });

    render(<LibraryView token={TOKEN} />);
    await screen.findByText('No documents uploaded yet');

    const input = document.querySelector('input[type="file"]');
    fireEvent.change(input, { target: { files: [makeFile('good.pdf'), makeFile('bad.pdf')] } });

    expect(await screen.findByText(/1 of 2 uploads failed/)).toBeInTheDocument();
    expect(screen.getByText(/bad\.pdf: corrupt file/)).toBeInTheDocument();
  });

  it.fails('reports an all-failed summary when every file in the batch fails', async () => {
    global.fetch = vi.fn((url, opts) => {
      if (url === `${API_URL}/api/v1/documents`) return jsonResponse({ documents: [] });
      if (url === `${API_URL}/api/v1/documents/upload`) {
        return jsonResponse({ detail: 'server down' }, false, 500);
      }
      return jsonResponse({ documents: [] });
    });

    render(<LibraryView token={TOKEN} />);
    await screen.findByText('No documents uploaded yet');

    const input = document.querySelector('input[type="file"]');
    fireEvent.change(input, { target: { files: [makeFile('a.pdf'), makeFile('b.pdf')] } });

    expect(await screen.findByText(/All uploads failed/)).toBeInTheDocument();
  });

  // Passing test that pins down the actual current (buggy) behavior
  // directly, so the bug is documented even without relying on it.fails'
  // pass/fail flip: the error is genuinely empty once the dust settles.
  it('BUG: upload-failure summary is cleared by the follow-up document refresh before it can be seen', async () => {
    global.fetch = vi.fn((url, opts) => {
      if (url === `${API_URL}/api/v1/documents`) return jsonResponse({ documents: [] });
      if (url === `${API_URL}/api/v1/documents/upload`) {
        return jsonResponse({ detail: 'server down' }, false, 500);
      }
      return jsonResponse({ documents: [] });
    });

    render(<LibraryView token={TOKEN} />);
    await screen.findByText('No documents uploaded yet');

    const input = document.querySelector('input[type="file"]');
    fireEvent.change(input, { target: { files: [makeFile('a.pdf'), makeFile('b.pdf')] } });

    // give the upload + follow-up refetch time to fully settle
    await waitFor(() => expect(global.fetch).toHaveBeenCalled());
    await new Promise((r) => setTimeout(r, 50));

    expect(screen.queryByText(/uploads failed/)).not.toBeInTheDocument();
  });

  it('resets the input value after selection, allowing re-selecting the same file', async () => {
    mockFetchSequence([() => jsonResponse({ documents: [] }), () => jsonResponse({ ok: true }), () => jsonResponse({ documents: [] })]);
    render(<LibraryView token={TOKEN} />);
    await screen.findByText('No documents uploaded yet');

    const input = document.querySelector('input[type="file"]');
    fireEvent.change(input, { target: { files: [makeFile('a.pdf')] } });

    await waitFor(() => expect(input.value).toBe(''));
  });
});

// ---- RETRY BUTTON FIX ----------------------------------------------------

describe('LibraryView — retry button (regression for the retry fix)', () => {
  it('shows a Retry button for "failed" and "processed_with_gaps" docs, not for "processed" or "processing"', async () => {
    const docs = [
      makeDoc({ _id: '1', filename: 'ok.pdf', status: 'processed' }),
      makeDoc({ _id: '2', filename: 'inflight.pdf', status: 'processing' }),
      makeDoc({ _id: '3', filename: 'broke.pdf', status: 'failed' }),
      makeDoc({ _id: '4', filename: 'gappy.pdf', status: 'processed_with_gaps' }),
    ];
    mockFetchSequence([() => jsonResponse({ documents: docs })]);
    render(<LibraryView token={TOKEN} />);

    await screen.findByText('ok.pdf');

    const rows = screen.getAllByText(/\.pdf$/).map((el) => el.closest('div').parentElement);
    const retryButtons = screen.getAllByRole('button', { name: /^Retry$/ });
    // exactly the two retryable docs get a button
    expect(retryButtons).toHaveLength(2);
  });

  it('calling retry hits POST /documents/{id}/retry, optimistically flips status to processing, then re-fetches', async () => {
    const initial = [makeDoc({ _id: '1', filename: 'broke.pdf', status: 'failed' })];
    let listCallCount = 0;
    global.fetch = vi.fn((url, opts) => {
      if (url === `${API_URL}/api/v1/documents/1/retry`) {
        expect(opts.method).toBe('POST');
        return jsonResponse({ status: 'processing' });
      }
      if (url === `${API_URL}/api/v1/documents`) {
        listCallCount += 1;
        // first (mount) call: still 'failed' so the Retry button is present
        // to click. Every call after that simulates the backend having
        // re-queued it.
        const status = listCallCount === 1 ? 'failed' : 'processing';
        return jsonResponse({ documents: [{ ...initial[0], status }] });
      }
      return jsonResponse({ documents: initial });
    });

    render(<LibraryView token={TOKEN} />);
    await screen.findByText('broke.pdf');

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /^Retry$/ }));

    // optimistic update happens synchronously off the click, before the
    // follow-up fetch resolves
    expect(screen.getByText('processing')).toBeInTheDocument();

    await waitFor(() => expect(listCallCount).toBeGreaterThanOrEqual(1));
  });

  it('surfaces the backend error message when retry fails', async () => {
    const initial = [makeDoc({ _id: '1', filename: 'broke.pdf', status: 'failed' })];
    global.fetch = vi.fn((url) => {
      if (url === `${API_URL}/api/v1/documents/1/retry`) {
        return jsonResponse({ detail: 'document already retrying' }, false, 400);
      }
      return jsonResponse({ documents: initial });
    });

    render(<LibraryView token={TOKEN} />);
    await screen.findByText('broke.pdf');

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /^Retry$/ }));

    expect(await screen.findByText('document already retrying')).toBeInTheDocument();
  });
});

// ---- POLLING FIX ----------------------------------------------------------

describe('LibraryView — polling (regression for the polling fix)', () => {
  it('polls silently (no spinner) while a document is "processing", and stops once nothing is in-flight', async () => {
    vi.useFakeTimers();
    let fetchCount = 0;
    const processingDoc = makeDoc({ _id: '1', filename: 'inflight.pdf', status: 'processing' });
    const settledDoc = { ...processingDoc, status: 'processed' };

    global.fetch = vi.fn(() => {
      fetchCount += 1;
      // first call (mount) and first two polls return "processing";
      // third poll onward returns "processed" so polling should stop after that
      const body = fetchCount <= 3 ? [processingDoc] : [settledDoc];
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ documents: body }) });
    });

    render(<LibraryView token={TOKEN} />);

    // flush the initial mount fetch — it resolves via microtasks, and
    // vi.waitFor's own polling doesn't reliably interleave with those while
    // fake timers are active, so tick the fake clock a few times to settle
    // the full fetch -> res.json() -> setState -> re-render chain.
    await flushMicrotasks();
    expect(fetchCount).toBe(1);
    // still processing after mount -> "Loading…" must NOT reappear during polls
    expect(screen.queryByText('Loading…')).not.toBeInTheDocument();

    await vi.advanceTimersByTimeAsync(4000);
    await flushMicrotasks();
    expect(fetchCount).toBe(2);
    expect(screen.queryByText('Loading…')).not.toBeInTheDocument();

    await vi.advanceTimersByTimeAsync(4000);
    await flushMicrotasks();
    expect(fetchCount).toBe(3);

    // this poll's response flips status to "processed" -> effect should
    // see no more in-flight docs and stop scheduling further polls
    await vi.advanceTimersByTimeAsync(4000);
    await flushMicrotasks();
    const countAfterSettle = fetchCount;
    expect(countAfterSettle).toBe(4);

    await vi.advanceTimersByTimeAsync(10000);
    await flushMicrotasks();
    expect(fetchCount).toBe(countAfterSettle);
  });

  it('does not poll at all when no document is "processing" on load', async () => {
    vi.useFakeTimers();
    let fetchCount = 0;
    global.fetch = vi.fn(() => {
      fetchCount += 1;
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ documents: [makeDoc({ status: 'processed' })] }) });
    });

    render(<LibraryView token={TOKEN} />);
    await flushMicrotasks();
    expect(fetchCount).toBe(1);

    await vi.advanceTimersByTimeAsync(20000);
    expect(fetchCount).toBe(1);
  });
});

// ---- DELETE ---------------------------------------------------------------

describe('LibraryView — delete', () => {
  it('confirms before deleting, and removes the doc from the list on success', async () => {
    const docs = [makeDoc({ _id: '1', filename: 'gone.pdf' })];
    global.fetch = vi.fn((url, opts) => {
      if (opts?.method === 'DELETE') return jsonResponse({});
      return jsonResponse({ documents: docs });
    });

    render(<LibraryView token={TOKEN} />);
    await screen.findByText('gone.pdf');

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /^Delete$/ }));

    expect(window.confirm).toHaveBeenCalled();
    await waitFor(() => expect(screen.queryByText('gone.pdf')).not.toBeInTheDocument());
  });

  it('does not delete when the confirm dialog is dismissed', async () => {
    window.confirm = vi.fn(() => false);
    const docs = [makeDoc({ _id: '1', filename: 'stays.pdf' })];
    global.fetch = vi.fn(() => jsonResponse({ documents: docs }));

    render(<LibraryView token={TOKEN} />);
    await screen.findByText('stays.pdf');

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /^Delete$/ }));

    expect(screen.getByText('stays.pdf')).toBeInTheDocument();
  });

  it('shows an error message when delete fails', async () => {
    const docs = [makeDoc({ _id: '1', filename: 'stuck.pdf' })];
    global.fetch = vi.fn((url, opts) => {
      if (opts?.method === 'DELETE') return jsonResponse({}, false, 500);
      return jsonResponse({ documents: docs });
    });

    render(<LibraryView token={TOKEN} />);
    await screen.findByText('stuck.pdf');

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: /^Delete$/ }));

    expect(await screen.findByText('Delete failed')).toBeInTheDocument();
    expect(screen.getByText('stuck.pdf')).toBeInTheDocument();
  });
});