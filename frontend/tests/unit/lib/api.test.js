import { describe, it, expect, beforeEach, vi } from "vitest";
import { apiRequest, apiGet, apiPost, apiDelete, apiUpload } from "../../../lib/api";

function mockFetchOnce({ ok, status = 200, statusText = "", jsonData = {}, jsonThrows = false }) {
  global.fetch = vi.fn().mockResolvedValue({
    ok,
    status,
    statusText,
    json: jsonThrows
      ? vi.fn().mockRejectedValue(new Error("body is not JSON"))
      : vi.fn().mockResolvedValue(jsonData),
  });
}

beforeEach(() => {
  localStorage.clear();
  vi.restoreAllMocks();
  Object.defineProperty(window, "location", {
    writable: true,
    value: { href: "" },
  });
});

describe("apiRequest - success paths", () => {
  it("calls fetch with Content-Type json and no Authorization header when no token stored", async () => {
    mockFetchOnce({ ok: true, jsonData: { result: "ok" } });

    const result = await apiRequest("/documents");

    const [url, options] = global.fetch.mock.calls[0];
    expect(url).toContain("/documents");
    expect(options.headers["Content-Type"]).toBe("application/json");
    expect(options.headers.Authorization).toBeUndefined();
    expect(result).toEqual({ result: "ok" });
  });

  it("attaches Authorization: Bearer <token> when a token is stored", async () => {
    localStorage.setItem("access_token", "my-token-123");
    mockFetchOnce({ ok: true, jsonData: {} });

    await apiRequest("/documents");

    const [, options] = global.fetch.mock.calls[0];
    expect(options.headers.Authorization).toBe("Bearer my-token-123");
  });

  it("merges caller-supplied headers on top of the defaults", async () => {
    mockFetchOnce({ ok: true, jsonData: {} });

    await apiRequest("/documents", { headers: { "X-Custom": "value" } });

    const [, options] = global.fetch.mock.calls[0];
    expect(options.headers["X-Custom"]).toBe("value");
    expect(options.headers["Content-Type"]).toBe("application/json");
  });
});

describe("apiRequest - 401 handling", () => {
  it("clears ONLY access_token (not user_id/user_name) and redirects to /auth/login on 401", async () => {
    localStorage.setItem("access_token", "expired-token");
    localStorage.setItem("user_id", "u1");
    localStorage.setItem("user_name", "Alice");
    mockFetchOnce({ ok: false, status: 401, statusText: "Unauthorized", jsonData: { detail: "Token expired" } });

    await expect(apiRequest("/documents")).rejects.toThrow("Token expired");

    expect(localStorage.getItem("access_token")).toBeNull();
    // KNOWN DIVERGENCE from apiClient.js: those interceptors also clear
    // user_id and user_name. api.js does not. Proven here, not assumed.
    expect(localStorage.getItem("user_id")).toBe("u1");
    expect(localStorage.getItem("user_name")).toBe("Alice");
    expect(window.location.href).toBe("/auth/login");
  });

  it("does NOT clear token or redirect on non-401 errors (e.g. 500)", async () => {
    localStorage.setItem("access_token", "still-valid-token");
    mockFetchOnce({ ok: false, status: 500, statusText: "Server Error", jsonData: { detail: "Internal error" } });

    await expect(apiRequest("/documents")).rejects.toThrow("Internal error");

    expect(localStorage.getItem("access_token")).toBe("still-valid-token");
    expect(window.location.href).toBe("");
  });
});

describe("apiRequest - error detail fallback", () => {
  it("uses backend's detail message when the error body is valid JSON", async () => {
    mockFetchOnce({ ok: false, status: 400, statusText: "Bad Request", jsonData: { detail: "Invalid file type" } });
    await expect(apiRequest("/documents")).rejects.toThrow("Invalid file type");
  });

  it("falls back to statusText when the error body isn't valid JSON", async () => {
    mockFetchOnce({ ok: false, status: 502, statusText: "Bad Gateway", jsonThrows: true });
    await expect(apiRequest("/documents")).rejects.toThrow("Bad Gateway");
  });
});

describe("HTTP verb wrappers", () => {
  it("apiGet issues a GET request", async () => {
    mockFetchOnce({ ok: true, jsonData: {} });
    await apiGet("/documents");
    expect(global.fetch.mock.calls[0][1].method).toBe("GET");
  });

  it("apiPost issues a POST with JSON-stringified body", async () => {
    mockFetchOnce({ ok: true, jsonData: {} });
    await apiPost("/documents", { title: "test" });
    const [, options] = global.fetch.mock.calls[0];
    expect(options.method).toBe("POST");
    expect(options.body).toBe(JSON.stringify({ title: "test" }));
  });

  it("apiDelete issues a DELETE request", async () => {
    mockFetchOnce({ ok: true, jsonData: {} });
    await apiDelete("/documents/1");
    expect(global.fetch.mock.calls[0][1].method).toBe("DELETE");
  });
});

describe("apiUpload", () => {
  function mockUploadFetchOnce({ ok, jsonData = {}, jsonThrows = false }) {
    global.fetch = vi.fn().mockResolvedValue({
      ok,
      json: jsonThrows
        ? vi.fn().mockRejectedValue(new Error("not json"))
        : vi.fn().mockResolvedValue(jsonData),
    });
  }

  it("sends a FormData body with the file, without a Content-Type header", async () => {
    mockUploadFetchOnce({ ok: true, jsonData: { id: "doc1" } });
    const file = new File(["content"], "test.pdf", { type: "application/pdf" });

    const result = await apiUpload("/documents/upload", file);

    const [, options] = global.fetch.mock.calls[0];
    expect(options.body).toBeInstanceOf(FormData);
    expect(options.headers["Content-Type"]).toBeUndefined();
    expect(result).toEqual({ id: "doc1" });
  });

  it("attaches Authorization header when a token is stored", async () => {
    localStorage.setItem("access_token", "upload-token");
    mockUploadFetchOnce({ ok: true, jsonData: {} });
    const file = new File(["x"], "a.pdf");

    await apiUpload("/documents/upload", file);

    expect(global.fetch.mock.calls[0][1].headers.Authorization).toBe("Bearer upload-token");
  });

  it("KNOWN DIVERGENCE: does NOT redirect on 401, unlike apiRequest", async () => {
    localStorage.setItem("access_token", "expired-token");
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      json: vi.fn().mockResolvedValue({ detail: "Token expired" }),
    });
    const file = new File(["x"], "a.pdf");

    await expect(apiUpload("/documents/upload", file)).rejects.toThrow("Token expired");

    // Unlike apiRequest's 401 branch, apiUpload never touches localStorage
    // or window.location — token stays put, no redirect happens.
    expect(localStorage.getItem("access_token")).toBe("expired-token");
    expect(window.location.href).toBe("");
  });

  it("KNOWN EDGE CASE: throws an unfriendly raw error if the error body isn't JSON (no try/catch fallback)", async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      json: vi.fn().mockRejectedValue(new Error("Unexpected token < in JSON")),
    });
    const file = new File(["x"], "a.pdf");

    // apiRequest would gracefully fall back to statusText here.
    // apiUpload has no such guard, so the raw JSON-parse error propagates.
    await expect(apiUpload("/documents/upload", file)).rejects.toThrow("Unexpected token < in JSON");
  });
});
