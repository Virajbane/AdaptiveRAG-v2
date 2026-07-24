import { describe, it, expect, beforeEach, vi } from "vitest";

const mockRequestUse = vi.fn();
const mockResponseUse = vi.fn();
const mockAxiosInstance = {
  interceptors: {
    request: { use: mockRequestUse },
    response: { use: mockResponseUse },
  },
};

vi.mock("axios", () => ({
  default: {
    create: vi.fn(() => mockAxiosInstance),
  },
}));

// Real helper (mirrors the one in security.test.js) so tests exercise the
// actual isTokenExpired() from config/security.js, not a stub of it.
function makeToken(payload) {
  const b64 = btoa(JSON.stringify(payload));
  return `header.${b64}.signature`;
}

let requestSuccessHandler;
let responseSuccessHandler;
let responseErrorHandler;

beforeEach(async () => {
  vi.resetModules();
  localStorage.clear();
  Object.defineProperty(window, "location", {
    writable: true,
    value: { href: "" },
  });

  await import("../../../lib/apiClient");

  requestSuccessHandler = mockRequestUse.mock.calls[0][0];
  responseSuccessHandler = mockResponseUse.mock.calls[0][0];
  responseErrorHandler = mockResponseUse.mock.calls[0][1];
});

describe("apiClient request interceptor", () => {
  it("passes config through unchanged when no token is stored", () => {
    const config = { headers: {} };
    const result = requestSuccessHandler(config);
    expect(result.headers.Authorization).toBeUndefined();
  });

  it("attaches Authorization: Bearer <token> for a valid, non-expired token", () => {
    const token = makeToken({ exp: Math.floor(Date.now() / 1000) + 3600 });
    localStorage.setItem("access_token", token);

    const config = { headers: {} };
    const result = requestSuccessHandler(config);

    expect(result.headers.Authorization).toBe(`Bearer ${token}`);
  });

  it("proactively clears access_token, user_id, AND user_name and redirects when the token is expired — BEFORE the request is even sent", async () => {
    const expiredToken = makeToken({ exp: Math.floor(Date.now() / 1000) - 3600 });
    localStorage.setItem("access_token", expiredToken);
    localStorage.setItem("user_id", "u1");
    localStorage.setItem("user_name", "Alice");

    const config = { headers: {} };

    await expect(requestSuccessHandler(config)).rejects.toThrow("Token expired");

    expect(localStorage.getItem("access_token")).toBeNull();
    expect(localStorage.getItem("user_id")).toBeNull();
    expect(localStorage.getItem("user_name")).toBeNull();
    expect(window.location.href).toBe("/auth/login?session_expired=true");
  });
});

describe("apiClient response interceptor", () => {
  it("passes successful responses through unchanged", () => {
    const response = { status: 200, data: { ok: true } };
    expect(responseSuccessHandler(response)).toBe(response);
  });

  it("on 401: clears all three storage keys and redirects to /auth/login?session_expired=true", async () => {
    localStorage.setItem("access_token", "sometoken");
    localStorage.setItem("user_id", "u1");
    localStorage.setItem("user_name", "Alice");

    const error = { response: { status: 401 } };

    await expect(responseErrorHandler(error)).rejects.toBe(error);

    expect(localStorage.getItem("access_token")).toBeNull();
    expect(localStorage.getItem("user_id")).toBeNull();
    expect(localStorage.getItem("user_name")).toBeNull();
    expect(window.location.href).toBe("/auth/login?session_expired=true");
  });

  it("on non-401 errors (e.g. 500): does NOT clear storage or redirect, still rejects with original error", async () => {
    localStorage.setItem("access_token", "sometoken");
    const error = { response: { status: 500 } };

    await expect(responseErrorHandler(error)).rejects.toBe(error);

    expect(localStorage.getItem("access_token")).toBe("sometoken");
    expect(window.location.href).toBe("");
  });

  it("on network errors with no response object at all: does not throw internally, still rejects", async () => {
    const error = { message: "Network Error" };
    await expect(responseErrorHandler(error)).rejects.toBe(error);
  });
});

describe("CROSS-FILE DIVERGENCE (apiClient.js vs lib/api.js)", () => {
  it("documents that apiClient clears 3 keys and uses a different redirect URL than api.js's apiRequest (which clears 1 key and redirects to /auth/login with no query param)", () => {
    // This test exists purely as a pointer between the two suites — see
    // tests/unit/lib/api.test.js "apiRequest - 401 handling" for the
    // api.js side of this same comparison. Nothing to assert here beyond
    // documenting where to look; the real proof lives in both suites.
    expect(true).toBe(true);
  });
});
