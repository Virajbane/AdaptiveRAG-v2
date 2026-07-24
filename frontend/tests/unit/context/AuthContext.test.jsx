import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { AuthProvider, useAuth } from "../../../app/context/AuthContext";

beforeEach(() => {
  localStorage.clear();
});

describe("AuthProvider - initial state", () => {
  it("starts unauthenticated with no token/user when localStorage is empty", () => {
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });

    expect(result.current.token).toBeNull();
    expect(result.current.user).toBeNull();
    expect(result.current.isAuthenticated).toBe(false);
  });

  it("restores token and user from localStorage on mount", () => {
    localStorage.setItem("access_token", "saved-token");
    localStorage.setItem("user", JSON.stringify({ id: "u1", name: "Alice", email: "a@x.com" }));

    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });

    expect(result.current.token).toBe("saved-token");
    expect(result.current.user).toEqual({ id: "u1", name: "Alice", email: "a@x.com" });
    expect(result.current.isAuthenticated).toBe(true);
  });

  it("stays unauthenticated if only one of token/user is present (both required)", () => {
    localStorage.setItem("access_token", "saved-token");
    // no 'user' key set

    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });

    expect(result.current.token).toBeNull();
    expect(result.current.isAuthenticated).toBe(false);
  });
});

describe("AuthProvider - login", () => {
  it("stores token and user in localStorage and updates context", () => {
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });

    act(() => {
      result.current.login("new-token", { id: "u2", name: "Bob" });
    });

    expect(result.current.token).toBe("new-token");
    expect(result.current.user).toEqual({ id: "u2", name: "Bob" });
    expect(result.current.isAuthenticated).toBe(true);
    expect(localStorage.getItem("access_token")).toBe("new-token");
    expect(JSON.parse(localStorage.getItem("user"))).toEqual({ id: "u2", name: "Bob" });
  });
});

describe("AuthProvider - logout", () => {
  it("clears access_token, user, and chat_messages; resets state to null", () => {
    localStorage.setItem("access_token", "tok");
    localStorage.setItem("user", JSON.stringify({ id: "u1" }));
    localStorage.setItem("chat_messages", JSON.stringify([{ role: "user", content: "hi" }]));

    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });

    act(() => {
      result.current.logout();
    });

    expect(result.current.token).toBeNull();
    expect(result.current.user).toBeNull();
    expect(result.current.isAuthenticated).toBe(false);
    expect(localStorage.getItem("access_token")).toBeNull();
    expect(localStorage.getItem("user")).toBeNull();
    expect(localStorage.getItem("chat_messages")).toBeNull();
  });

  it("KNOWN GAP: logout() does not clear ChatContext's actual storage key (ragworkspace:activeChat) — only the unrelated 'chat_messages' key", () => {
    localStorage.setItem("access_token", "tok");
    localStorage.setItem("user", JSON.stringify({ id: "u1" }));
    // This is the REAL key ChatContext.jsx persists to (STORAGE_KEY constant),
    // not 'chat_messages'. logout() has no idea this key exists.
    localStorage.setItem("ragworkspace:activeChat", JSON.stringify({ messages: [{ role: "user", content: "hi" }] }));

    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });

    act(() => {
      result.current.logout();
    });

    // Proves the gap: chat history survives logout, visible to whoever
    // logs in next on the same browser.
    expect(localStorage.getItem("ragworkspace:activeChat")).not.toBeNull();
  });
});

describe("useAuth - misuse", () => {
  it("throws a clear error when used outside AuthProvider", () => {
    // Suppress the expected console.error React logs for this thrown-render test
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => renderHook(() => useAuth())).toThrow(
      "useAuth must be used within AuthProvider"
    );
    spy.mockRestore();
  });
});

describe("AuthProvider - KNOWN BUG: unguarded JSON.parse on corrupted localStorage", () => {
  it("crashes (throws, uncaught) when the stored 'user' value is not valid JSON", () => {
    localStorage.setItem("access_token", "tok");
    localStorage.setItem("user", "{this is not valid json");

    const spy = vi.spyOn(console, "error").mockImplementation(() => {});

    // This test PASSES because the crash currently happens — it documents
    // the real, present behavior described in the audit log (no try/catch
    // around JSON.parse). If this is ever fixed with a guard, this test
    // should start failing, which is the correct signal to update it to
    // assert graceful fallback behavior instead.
    expect(() => renderHook(() => useAuth(), { wrapper: AuthProvider })).toThrow();

    spy.mockRestore();
  });
});
