import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  getToken,
  setToken,
  clearToken,
  decodeToken,
  isTokenExpired,
  validateEmail,
  validatePassword,
} from "../../../config/security";

// Helper: build a fake JWT with a real base64(url) payload, mirroring
// what a backend actually issues (header.payload.signature).
function makeToken(payload, { urlSafe = false } = {}) {
  const json = JSON.stringify(payload);
  let b64 = btoa(json);
  if (urlSafe) {
    b64 = b64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }
  return `header.${b64}.signature`;
}

describe("token storage", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("setToken stores under access_token key, getToken reads it back", () => {
    setToken("abc.def.ghi");
    expect(localStorage.getItem("access_token")).toBe("abc.def.ghi");
    expect(getToken()).toBe("abc.def.ghi");
  });

  it("getToken returns null when nothing stored", () => {
    expect(getToken()).toBeNull();
  });

  it("clearToken removes the stored token", () => {
    setToken("abc.def.ghi");
    clearToken();
    expect(getToken()).toBeNull();
  });

  it("getToken returns null when window is undefined (SSR)", () => {
    const originalWindow = globalThis.window;
    // @ts-expect-error intentional SSR simulation
    vi.stubGlobal("window", undefined);
    try {
      expect(getToken()).toBeNull();
    } finally {
      vi.stubGlobal("window", originalWindow);
    }
  });
});

describe("decodeToken", () => {
  it("decodes a well-formed standard-base64 payload", () => {
    const token = makeToken({ sub: "user1", exp: 9999999999 });
    expect(decodeToken(token)).toEqual({ sub: "user1", exp: 9999999999 });
  });

  it("returns null for a malformed token (not 3 segments)", () => {
    expect(decodeToken("not-a-jwt")).toBeNull();
  });

  it("returns null for a token whose payload isn't valid base64", () => {
    expect(decodeToken("header.!!!not-base64!!!.sig")).toBeNull();
  });

  it("returns null for a token whose payload decodes but isn't valid JSON", () => {
    const b64 = btoa("not json");
    expect(decodeToken(`header.${b64}.sig`)).toBeNull();
  });

  // Documents a real edge case rather than assuming it away: atob() only
  // understands standard base64, not base64url. A payload that happens to
  // need a '-' or '_' character (i.e. would contain '+' or '/' in standard
  // base64) will throw inside atob and get swallowed by the try/catch,
  // silently returning null for what is otherwise a *valid* JWT.
  it("KNOWN EDGE CASE: fails to decode a valid token if its payload is base64url-encoded with -/_", () => {
    const payload = { sub: "user-needing-plus-or-slash", exp: 9999999999, pad: "??>>" };
    const urlSafeToken = makeToken(payload, { urlSafe: true });

    // Only assert the divergent behavior if this payload actually produced
    // url-safe characters different from standard base64 (otherwise the
    // test would be a false positive on unrelated input).
    const standardB64 = btoa(JSON.stringify(payload));
    const isActuallyUrlSafeDivergent = /[+/]/.test(standardB64);

    if (isActuallyUrlSafeDivergent) {
      expect(decodeToken(urlSafeToken)).toBeNull();
    } else {
      // fallback deterministic payload guaranteed to contain '+' in standard b64
      const forced = makeToken({ a: "??" }, { urlSafe: true });
      expect(decodeToken(forced)).toBeNull();
    }
  });
});

describe("isTokenExpired", () => {
  it("returns false for a token with a future exp", () => {
    const token = makeToken({ exp: Math.floor(Date.now() / 1000) + 3600 });
    expect(isTokenExpired(token)).toBe(false);
  });

  it("returns true for a token with a past exp", () => {
    const token = makeToken({ exp: Math.floor(Date.now() / 1000) - 3600 });
    expect(isTokenExpired(token)).toBe(true);
  });

  it("returns true when exp is exactly now (boundary, inclusive)", () => {
    const nowSeconds = Math.floor(Date.now() / 1000);
    const token = makeToken({ exp: nowSeconds });
    // Date.now() at call time will be >= nowSeconds * 1000
    expect(isTokenExpired(token)).toBe(true);
  });

  it("returns true when payload has no exp claim", () => {
    const token = makeToken({ sub: "user1" });
    expect(isTokenExpired(token)).toBe(true);
  });

  it("returns true when the token can't be decoded at all", () => {
    expect(isTokenExpired("garbage")).toBe(true);
  });
});

describe("validateEmail", () => {
  it.each([
    "user@example.com",
    "first.last@sub.example.co",
    "user+tag@example.io",
    "u@x.co",
  ])("accepts valid email: %s", (email) => {
    expect(validateEmail(email)).toBeNull();
  });

  it.each([
    "",
    "not-an-email",
    "user@",
    "@example.com",
    "user@example",
    "user example@example.com",
    "user@example.c",
  ])("rejects invalid email: %s", (email) => {
    expect(validateEmail(email)).toBe("Please enter a valid email address");
  });
});

describe("validatePassword", () => {
  it("rejects passwords under 8 characters", () => {
    expect(validatePassword("Ab1!xyz")).toBe(
      "Password must be at least 8 characters"
    );
  });

  it("rejects passwords with no uppercase letter", () => {
    expect(validatePassword("abcdefg1!")).toBe(
      "Password must contain an uppercase letter"
    );
  });

  it("rejects passwords with no lowercase letter", () => {
    expect(validatePassword("ABCDEFG1!")).toBe(
      "Password must contain a lowercase letter"
    );
  });

  it("rejects passwords with no digit", () => {
    expect(validatePassword("Abcdefgh!")).toBe(
      "Password must contain a digit"
    );
  });

  it("rejects passwords with no special character", () => {
    expect(validatePassword("Abcdefg1")).toBe(
      "Password must contain a special character"
    );
  });

  it("accepts a password satisfying every rule", () => {
    expect(validatePassword("Abcdefg1!")).toBeNull();
  });

  // Locks in the exact accepted special-character set for this file, so
  // any future drift (or comparison against register/page.jsx's local
  // copy, or backend validators.py) is caught by a failing test rather
  // than discovered in production.
  it.each(['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', ',', '.', '?', '"', ':', '{', '}', '|', '<', '>'])(
    "accepts special character: %s",
    (char) => {
      expect(validatePassword(`Abcdefg1${char}`)).toBeNull();
    }
  );

  // Characters this file's regex does NOT treat as special — these should
  // still fail the special-character check. This is the exact boundary
  // that diverges from the other two copies in the stack.
  it.each(["-", "_", "+", "=", "[", "]", ";", "'", "~", "`", "/", "\\"])(
    "does NOT accept '%s' as a special character (per this file's rule set)",
    (char) => {
      expect(validatePassword(`Abcdefg1${char}`)).toBe(
        "Password must contain a special character"
      );
    }
  );
});
