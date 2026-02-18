import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { createRequire } from "node:module";
import { TruseraClient } from "../src/client.js";
import { TruseraInterceptor } from "../src/interceptor.js";

// Use CJS require to get the same undici instance as the interceptor
const require_ = createRequire(import.meta.url);
const undici = require_("undici");

// Save original functions
const origRequest = undici.request;
const origFetch = undici.fetch;
const originalGlobalFetch = globalThis.fetch;

let client: TruseraClient;
let interceptor: TruseraInterceptor;

beforeEach(() => {
  client = new TruseraClient({
    apiKey: "tsk_test123",
    flushInterval: 999999,
  });
  interceptor = new TruseraInterceptor();
});

afterEach(() => {
  interceptor.uninstall();
  globalThis.fetch = originalGlobalFetch;
  // Restore undici originals as safety net
  undici.request = origRequest;
  undici.fetch = origFetch;
});

describe("Undici Interception", () => {
  describe("undici.request", () => {
    it("should patch undici.request on install", () => {
      interceptor.install(client);
      expect(undici.request).not.toBe(origRequest);
    });

    it("should restore original undici.request on uninstall", () => {
      interceptor.install(client);
      interceptor.uninstall();
      expect(undici.request).toBe(origRequest);
    });
  });

  describe("undici.fetch", () => {
    it("should patch undici.fetch on install", () => {
      interceptor.install(client);
      expect(undici.fetch).not.toBe(origFetch);
    });

    it("should restore original undici.fetch on uninstall", () => {
      interceptor.install(client);
      interceptor.uninstall();
      expect(undici.fetch).toBe(origFetch);
    });
  });

  describe("lifecycle", () => {
    it("should survive multiple install/uninstall cycles", () => {
      for (let i = 0; i < 3; i++) {
        interceptor.install(client);
        expect(undici.request).not.toBe(origRequest);
        expect(undici.fetch).not.toBe(origFetch);
        interceptor.uninstall();
        expect(undici.request).toBe(origRequest);
        expect(undici.fetch).toBe(origFetch);
      }
    });

    it("should handle exclude patterns", () => {
      interceptor.install(client, {
        excludePatterns: ["^https://api\\.trusera\\.io"],
      });

      // Patched function exists
      expect(undici.request).not.toBe(origRequest);
      expect(undici.fetch).not.toBe(origFetch);

      interceptor.uninstall();
    });
  });
});
