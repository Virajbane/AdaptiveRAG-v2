// tests/unit/components/ChatView.test.jsx
//
// ASSUMPTION: source path is `app/components/layout/dashboard/views/ChatView.jsx`
// — mirroring where LibraryView.jsx turned out to actually live
// (`components/layout/dashboard/views/`). Adjust the import below if wrong.
//
// NOTE ON THE @/ ALIAS: ChatView.jsx itself imports `useChat` via the `@/`
// alias (`@/app/context/ChatContext`), even though this project's tests use
// relative imports elsewhere because that alias doesn't resolve reliably
// under Vitest. We don't need the alias to resolve here, though: `vi.mock`
// intercepts by the exact specifier string as written in the file being
// tested, before Vite attempts real resolution — so mocking
// '@/app/context/ChatContext' below should short-circuit that broken
// resolution entirely. If this suite fails with an "Failed to resolve
// import" error (the same class of error the LibraryView suite hit), that's
// the alias breaking through instead of being intercepted, and the fix is
// to switch ChatView.jsx's own import to a relative path.

import ChatView from '../../../components/layout/dashboard/views/ChatView';
import { useChat } from '@/app/context/ChatContext';

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

vi.mock('@/app/context/ChatContext', () => ({
  useChat: vi.fn(),
}));

// ---- helpers -------------------------------------------------------------

function baseChatState(overrides = {}) {
  return {
    messages: [],
    loading: false,
    elapsed: 0,
    sendMessage: vi.fn(),
    cancelMessage: vi.fn(),
    ...overrides,
  };
}

function makeMessage(overrides = {}) {
  return {
    role: 'assistant',
    content: 'Here is the answer.',
    ...overrides,
  };
}

beforeEach(() => {
  // jsdom doesn't implement these; ChatView calls both.
  Element.prototype.scrollIntoView = vi.fn();
  global.requestAnimationFrame = vi.fn((cb) => { cb(); return 0; });
  useChat.mockReturnValue(baseChatState());
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// ---- empty state ----------------------------------------------------------

describe('ChatView — empty state', () => {
  it('shows the empty prompt when there are no messages', () => {
    render(<ChatView documentCount={3} />);
    expect(screen.getByText('Ask me anything about your documents.')).toBeInTheDocument();
  });

  it('pluralizes "file(s)" correctly for 0, 1, and multiple documents', () => {
    const { rerender } = render(<ChatView documentCount={0} />);
    expect(screen.getByText(/Searching 0 files/)).toBeInTheDocument();

    rerender(<ChatView documentCount={1} />);
    expect(screen.getByText(/Searching 1 file(?!s)/)).toBeInTheDocument();

    rerender(<ChatView documentCount={5} />);
    expect(screen.getByText(/Searching 5 files/)).toBeInTheDocument();
  });

  it('hides the empty prompt once there is at least one message', () => {
    useChat.mockReturnValue(baseChatState({ messages: [makeMessage()] }));
    render(<ChatView documentCount={2} />);
    expect(screen.queryByText('Ask me anything about your documents.')).not.toBeInTheDocument();
  });
});

// ---- message rendering ----------------------------------------------------

describe('ChatView — message rendering', () => {
  it('renders user and assistant message content', () => {
    useChat.mockReturnValue(baseChatState({
      messages: [
        makeMessage({ role: 'user', content: 'What does the contract say about termination?' }),
        makeMessage({ role: 'assistant', content: 'Section 4.2 covers termination.' }),
      ],
    }));
    render(<ChatView />);
    expect(screen.getByText('What does the contract say about termination?')).toBeInTheDocument();
    expect(screen.getByText('Section 4.2 covers termination.')).toBeInTheDocument();
    expect(screen.getByText('AI Assistant')).toBeInTheDocument();
  });

  it('renders a sources-cited card for each source, only when sources exist', () => {
    useChat.mockReturnValue(baseChatState({
      messages: [
        makeMessage({
          sources: [
            { filename: 'handbook.pdf', page: 12, score: 0.87 },
            { filename: 'policy.pdf' },
          ],
        }),
      ],
    }));
    render(<ChatView />);
    expect(screen.getByText('Sources Cited')).toBeInTheDocument();
    expect(screen.getByText('handbook.pdf')).toBeInTheDocument();
    expect(screen.getByText(/Page 12/)).toBeInTheDocument();
    expect(screen.getByText(/87% match/)).toBeInTheDocument();
    expect(screen.getByText('policy.pdf')).toBeInTheDocument();
  });

  it('omits the sources section entirely when a message has no sources', () => {
    useChat.mockReturnValue(baseChatState({ messages: [makeMessage({ sources: [] })] }));
    render(<ChatView />);
    expect(screen.queryByText('Sources Cited')).not.toBeInTheDocument();
  });

  it('renders confidence and search time when present, and omits search time when absent', () => {
    useChat.mockReturnValue(baseChatState({
      messages: [makeMessage({ confidence: 0.923, searchTimeMs: 481.6 })],
    }));
    const { rerender } = render(<ChatView />);
    expect(screen.getByText(/Confidence: 92%/)).toBeInTheDocument();
    expect(screen.getByText(/482ms/)).toBeInTheDocument();

    useChat.mockReturnValue(baseChatState({
      messages: [makeMessage({ confidence: 0.5 })],
    }));
    rerender(<ChatView />);
    expect(screen.getByText('Confidence: 50%')).toBeInTheDocument();
  });

  it('scrolls the message thread into view whenever messages change', () => {
    const { rerender } = render(<ChatView />);
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();

    Element.prototype.scrollIntoView.mockClear();
    useChat.mockReturnValue(baseChatState({ messages: [makeMessage()] }));
    rerender(<ChatView />);
    expect(Element.prototype.scrollIntoView).toHaveBeenCalledWith({ behavior: 'smooth' });
  });
});

// ---- composer: sending -----------------------------------------------------

describe('ChatView — composer send behavior', () => {
  it('sends on Enter and clears the input', async () => {
    const sendMessage = vi.fn();
    useChat.mockReturnValue(baseChatState({ sendMessage }));
    render(<ChatView />);

    const textarea = screen.getByPlaceholderText('Ask about your knowledge base…');
    const user = userEvent.setup();
    await user.type(textarea, 'What are the payment terms?');
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(sendMessage).toHaveBeenCalledWith('What are the payment terms?');
    expect(textarea).toHaveValue('');
  });

  it('does not send on Shift+Enter (newline instead)', async () => {
    const sendMessage = vi.fn();
    useChat.mockReturnValue(baseChatState({ sendMessage }));
    render(<ChatView />);

    const textarea = screen.getByPlaceholderText('Ask about your knowledge base…');
    const user = userEvent.setup();
    await user.type(textarea, 'line one');
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true });

    expect(sendMessage).not.toHaveBeenCalled();
  });

  it('does not send whitespace-only input', async () => {
    const sendMessage = vi.fn();
    useChat.mockReturnValue(baseChatState({ sendMessage }));
    render(<ChatView />);

    const textarea = screen.getByPlaceholderText('Ask about your knowledge base…');
    const user = userEvent.setup();
    await user.type(textarea, '   ');
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(sendMessage).not.toHaveBeenCalled();
  });

  it('sends the raw (untrimmed) input value, not a trimmed copy', async () => {
    // handleSend only uses input.trim() to decide *whether* to send — it
    // still passes the original `input` straight through to sendMessage.
    const sendMessage = vi.fn();
    useChat.mockReturnValue(baseChatState({ sendMessage }));
    render(<ChatView />);

    const textarea = screen.getByPlaceholderText('Ask about your knowledge base…');
    const user = userEvent.setup();
    await user.type(textarea, '  padded question  ');
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false });

    expect(sendMessage).toHaveBeenCalledWith('  padded question  ');
  });

  it('disables the send button when input is empty and enables it once text is entered', async () => {
    render(<ChatView />);
    const textarea = screen.getByPlaceholderText('Ask about your knowledge base…');
    // DOM order with no messages rendered: [+ button, attach button, send
    // button, @web tag, @sql tag]. None of these buttons have distinguishing
    // accessible names (icon-only), so we pin down the send button by its
    // known position rather than an ambiguous CSS selector.
    const sendButton = screen.getAllByRole('button')[2];

    expect(sendButton).toBeDisabled();
    const user = userEvent.setup();
    await user.type(textarea, 'hi');
    expect(sendButton).not.toBeDisabled();
  });

  it('disables the textarea and send button while loading', () => {
    useChat.mockReturnValue(baseChatState({ loading: true }));
    render(<ChatView />);
    const textarea = screen.getByPlaceholderText('Ask about your knowledge base…');
    expect(textarea).toBeDisabled();
    expect(screen.getAllByRole('button')[2]).toBeDisabled();
  });
});

// ---- loading / cancel / the elapsed bug -----------------------------------

describe('ChatView — loading indicator and cancel', () => {
  it('shows the typing indicator and a cancel button while loading', () => {
    useChat.mockReturnValue(baseChatState({ loading: true, elapsed: 4 }));
    render(<ChatView />);
    expect(screen.getByText('Cancel (4s)')).toBeInTheDocument();
  });

  it('calls cancelMessage when the cancel button is clicked', async () => {
    const cancelMessage = vi.fn();
    useChat.mockReturnValue(baseChatState({ loading: true, elapsed: 2, cancelMessage }));
    render(<ChatView />);

    const user = userEvent.setup();
    await user.click(screen.getByText(/^Cancel/));
    expect(cancelMessage).toHaveBeenCalled();
  });

  it('hides the typing indicator once loading is false', () => {
    useChat.mockReturnValue(baseChatState({ loading: false }));
    render(<ChatView />);
    expect(screen.queryByText(/^Cancel/)).not.toBeInTheDocument();
  });

  // Ties back to the Stage 3 `it.fails` tripwire in ChatContext.test.jsx
  // (useChat() doesn't expose `elapsed`). This test mocks useChat() to
  // match that *actual* real-world shape — no `elapsed` key at all — and
  // confirms the bug flows all the way through to the rendered UI: the
  // cancel button literally reads "Cancel (undefineds)". This is a
  // passing test that documents current (buggy) behavior, per the
  // project's ground rules; it should be revisited (and likely deleted or
  // flipped) the moment the Stage 3 elapsed fix lands.
  it('BUG (tied to Stage 3 elapsed finding): renders literal "Cancel (undefineds)" when useChat omits elapsed', () => {
    useChat.mockReturnValue({
      messages: [],
      loading: true,
      // no `elapsed` key — matches real ChatContext's actual current shape
      sendMessage: vi.fn(),
      cancelMessage: vi.fn(),
    });
    render(<ChatView />);
    expect(screen.getByText('Cancel (undefineds)')).toBeInTheDocument();
  });
});

// ---- tag shortcuts ----------------------------------------------------------

describe('ChatView — tag shortcut buttons', () => {
  it('prepends the tag to an empty composer', async () => {
    render(<ChatView />);
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: '@WEB' }));

    const textarea = screen.getByPlaceholderText('Ask about your knowledge base…');
    expect(textarea).toHaveValue('@web ');
  });

  it('prepends the tag in front of existing text', async () => {
    render(<ChatView />);
    const textarea = screen.getByPlaceholderText('Ask about your knowledge base…');
    const user = userEvent.setup();
    await user.type(textarea, 'refund policy');
    await user.click(screen.getByRole('button', { name: '@SQL' }));

    expect(textarea).toHaveValue('@sql refund policy');
  });

  it('does not duplicate a tag already present within the first 3 words', async () => {
    render(<ChatView />);
    const textarea = screen.getByPlaceholderText('Ask about your knowledge base…');
    const user = userEvent.setup();
    await user.type(textarea, '@web refund policy');
    await user.click(screen.getByRole('button', { name: '@WEB' }));

    expect(textarea).toHaveValue('@web refund policy');
  });

  it('DOCUMENTED LIMITATION: does duplicate the tag if it appears at word index 3 or later', async () => {
    // handleTagClick's "already present" check only inspects the first 3
    // words (a deliberate simplification per the component's own comment,
    // not a re-implementation of the backend's leading-tag parser) — so a
    // tag mentioned later in the message isn't recognized as already
    // present and gets prepended again.
    render(<ChatView />);
    const textarea = screen.getByPlaceholderText('Ask about your knowledge base…');
    const user = userEvent.setup();
    await user.type(textarea, 'one two three @web');
    await user.click(screen.getByRole('button', { name: '@WEB' }));

    expect(textarea).toHaveValue('@web one two three @web');
  });

  it('shows the tooltip on hover and hides it on mouse leave', async () => {
    render(<ChatView />);
    const webButton = screen.getByRole('button', { name: '@WEB' });

    expect(screen.queryByText(/Searches the web/)).not.toBeInTheDocument();
    fireEvent.mouseEnter(webButton);
    expect(screen.getByText(/Searches the web/)).toBeInTheDocument();
    fireEvent.mouseLeave(webButton);
    expect(screen.queryByText(/Searches the web/)).not.toBeInTheDocument();
  });
});