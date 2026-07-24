import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { ChatProvider, useChat } from "../../../app/context/ChatContext";

const STORAGE_KEY = "ragworkspace:activeChat";

function wrapper({ children }) {
  return <ChatProvider token="test-token">{children}</ChatProvider>;
}

function mockAbortableFetch({ ok = true, status = 200, jsonValue = {}, jsonThrows = false, delay = 20 } = {}) {
  return vi.fn((url, options) => {
    return new Promise((resolve, reject) => {
      const signal = options?.signal;
      const timer = setTimeout(() => {
        resolve({
          ok,
          status,
          json: jsonThrows
            ? vi.fn().mockRejectedValue(new Error("not json"))
            : vi.fn().mockResolvedValue(jsonValue),
        });
      }, delay);

      if (signal) {
        signal.addEventListener("abort", () => {
          clearTimeout(timer);
          const err = new Error("The operation was aborted");
          err.name = "AbortError";
          reject(err);
        });
      }
    });
  });
}

beforeEach(() => {
  localStorage.clear();
});

describe("ChatProvider - initial load from storage", () => {
  it("starts with an empty message list when nothing is stored", () => {
    const { result } = renderHook(() => useChat(), { wrapper });
    expect(result.current.messages).toEqual([]);
  });

  it("restores messages from localStorage on mount", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ messages: [{ role: "user", content: "hi" }], savedAt: Date.now() })
    );
    const { result } = renderHook(() => useChat(), { wrapper });
    expect(result.current.messages).toEqual([{ role: "user", content: "hi" }]);
  });

  it("falls back to [] if stored data has no .messages property", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ savedAt: Date.now() }));
    const { result } = renderHook(() => useChat(), { wrapper });
    expect(result.current.messages).toEqual([]);
  });

  it("CONTRAST vs AuthContext: gracefully falls back to [] (no crash) when stored JSON is corrupted, because this file DOES wrap its parse in try/catch", () => {
    localStorage.setItem(STORAGE_KEY, "{this is not valid json");
    expect(() => renderHook(() => useChat(), { wrapper })).not.toThrow();
    const { result } = renderHook(() => useChat(), { wrapper });
    expect(result.current.messages).toEqual([]);
  });
});

describe("ChatProvider - sendMessage guards", () => {
  it("does nothing for an empty string", async () => {
    const { result } = renderHook(() => useChat(), { wrapper });
    await act(async () => {
      await result.current.sendMessage("");
    });
    expect(result.current.messages).toEqual([]);
  });

  it("does nothing for a whitespace-only string", async () => {
    const { result } = renderHook(() => useChat(), { wrapper });
    await act(async () => {
      await result.current.sendMessage("   ");
    });
    expect(result.current.messages).toEqual([]);
  });

  it("ignores a second sendMessage call while the first is still loading", async () => {
    global.fetch = mockAbortableFetch({ jsonValue: { answer: "reply" }, delay: 100 });
    const { result } = renderHook(() => useChat(), { wrapper });

    act(() => {
      result.current.sendMessage("first");
    });
    expect(result.current.loading).toBe(true);
    const countAfterFirstCall = result.current.messages.length;

    act(() => {
      result.current.sendMessage("second");
    });
    expect(result.current.messages.length).toBe(countAfterFirstCall);

    await act(async () => {
      await new Promise((r) => setTimeout(r, 150));
    });
    expect(result.current.loading).toBe(false);
    expect(result.current.messages.filter((m) => m.role === "user")).toEqual([
      { role: "user", content: "first" },
    ]);
  });
});

describe("ChatProvider - success path", () => {
  it("appends the user message immediately, then the assistant reply with confidence/sources/searchTimeMs", async () => {
    global.fetch = mockAbortableFetch({
      jsonValue: { answer: "The answer", confidence: 0.92, sources: [{ doc_id: "d1" }], search_time_ms: 123 },
      delay: 10,
    });
    const { result } = renderHook(() => useChat(), { wrapper });

    await act(async () => {
      await result.current.sendMessage("What is X?");
    });

    expect(result.current.messages).toEqual([
      { role: "user", content: "What is X?" },
      {
        role: "assistant",
        content: "The answer",
        confidence: 0.92,
        sources: [{ doc_id: "d1" }],
        searchTimeMs: 123,
      },
    ]);
    expect(result.current.loading).toBe(false);
  });

  it("defaults sources to [] when the backend omits them", async () => {
    global.fetch = mockAbortableFetch({ jsonValue: { answer: "ok" }, delay: 10 });
    const { result } = renderHook(() => useChat(), { wrapper });

    await act(async () => {
      await result.current.sendMessage("hello");
    });

    expect(result.current.messages[1].sources).toEqual([]);
  });
});

describe("ChatProvider - error path", () => {
  it("uses the backend's detail message on a non-ok JSON error response", async () => {
    global.fetch = mockAbortableFetch({ ok: false, status: 400, jsonValue: { detail: "Bad input" }, delay: 10 });
    const { result } = renderHook(() => useChat(), { wrapper });

    await act(async () => {
      await result.current.sendMessage("hello");
    });

    expect(result.current.messages[1]).toEqual({ role: "assistant", content: "Error: Bad input" });
  });

  it("falls back to 'Error {status}' when the error body isn't JSON", async () => {
    global.fetch = mockAbortableFetch({ ok: false, status: 500, jsonThrows: true, delay: 10 });
    const { result } = renderHook(() => useChat(), { wrapper });

    await act(async () => {
      await result.current.sendMessage("hello");
    });

    expect(result.current.messages[1]).toEqual({ role: "assistant", content: "Error: Error 500" });
  });
});

describe("ChatProvider - cancelMessage / timeout", () => {
  it("cancelMessage() aborts the in-flight request and shows the 15-minute timeout message", async () => {
    global.fetch = mockAbortableFetch({ jsonValue: { answer: "too late" }, delay: 5000 });
    const { result } = renderHook(() => useChat(), { wrapper });

    await act(async () => {
      const sendPromise = result.current.sendMessage("hello");
      await Promise.resolve();
      result.current.cancelMessage();
      await sendPromise;
    });

    expect(result.current.messages[1]).toEqual({
      role: "assistant",
      content: "Request timed out after 15 minutes.",
    });
    expect(result.current.loading).toBe(false);
  });
});

describe("ChatProvider - persistence", () => {
  it("writes messages to localStorage under the real STORAGE_KEY after every change", async () => {
    global.fetch = mockAbortableFetch({ jsonValue: { answer: "ok" }, delay: 10 });
    const { result } = renderHook(() => useChat(), { wrapper });

    await act(async () => {
      await result.current.sendMessage("hello");
    });

    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    expect(saved.messages).toEqual(result.current.messages);
    expect(typeof saved.savedAt).toBe("number");
  });
});

describe("ChatProvider - newChat", () => {
  it("resets messages to [] and loading to false", async () => {
    global.fetch = mockAbortableFetch({ jsonValue: { answer: "ok" }, delay: 10 });
    const { result } = renderHook(() => useChat(), { wrapper });

    await act(async () => {
      await result.current.sendMessage("hello");
    });
    expect(result.current.messages.length).toBeGreaterThan(0);

    act(() => {
      result.current.newChat();
    });

    expect(result.current.messages).toEqual([]);
    expect(result.current.loading).toBe(false);
  });

  it("REAL BUG FOUND: newChat()'s localStorage.removeItem() is immediately undone by the persistence effect — the key is never actually absent, only reset to an empty-array shape", async () => {
    global.fetch = mockAbortableFetch({ jsonValue: { answer: "ok" }, delay: 10 });
    const { result } = renderHook(() => useChat(), { wrapper });

    await act(async () => {
      await result.current.sendMessage("hello");
    });
    expect(localStorage.getItem(STORAGE_KEY)).not.toBeNull();

    act(() => {
      result.current.newChat();
    });

    // What the code comment implies should happen ("ONLY this clears the
    // chat") — it does NOT: the persistence useEffect runs on the very
    // next commit (messages changed to []) and re-writes the key with
    // { messages: [], savedAt: ... }. removeItem() executes synchronously
    // inside newChat(), but the effect fires afterward and wins.
    const afterNewChat = localStorage.getItem(STORAGE_KEY);
    expect(afterNewChat).not.toBeNull();
    expect(JSON.parse(afterNewChat).messages).toEqual([]);
  });
});

describe("useChat - misuse", () => {
  it("throws a clear error when used outside ChatProvider", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => renderHook(() => useChat())).toThrow("useChat must be used inside ChatProvider");
    spy.mockRestore();
  });
});

describe("ChatProvider - DOCUMENTED BUG: missing 'elapsed'", () => {
  it.fails("exposes 'elapsed' from useChat() (currently does not — ChatView.jsx reads it as undefined)", () => {
    const { result } = renderHook(() => useChat(), { wrapper });
    expect(result.current.elapsed).toBeDefined();
  });
});
