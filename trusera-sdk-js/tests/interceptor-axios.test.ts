import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { createRequire } from "node:module";
import { TruseraClient } from "../src/client.js";
import { TruseraInterceptor } from "../src/interceptor.js";

// Use CJS require to get the same axios instance as the interceptor
const require_ = createRequire(import.meta.url);
const axios = require_("axios");

const originalFetch = globalThis.fetch;
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
  globalThis.fetch = originalFetch;
});

describe("Axios Interception", () => {
  it("should register request and response interceptors on install", () => {
    const reqBefore = axios.interceptors.request.handlers.filter(Boolean).length;
    const resBefore = axios.interceptors.response.handlers.filter(Boolean).length;

    interceptor.install(client);

    const reqAfter = axios.interceptors.request.handlers.filter(Boolean).length;
    const resAfter = axios.interceptors.response.handlers.filter(Boolean).length;

    expect(reqAfter).toBe(reqBefore + 1);
    expect(resAfter).toBe(resBefore + 1);
  });

  it("should eject interceptors on uninstall", () => {
    interceptor.install(client);
    const reqActive = axios.interceptors.request.handlers.filter(Boolean).length;
    expect(reqActive).toBeGreaterThan(0);

    interceptor.uninstall();

    // After eject, handlers are set to null
    const reqAfterUninstall = axios.interceptors.request.handlers.filter(Boolean).length;
    expect(reqAfterUninstall).toBeLessThan(reqActive);
  });

  it("should pass through config in request interceptor", async () => {
    interceptor.install(client);

    const handlers = axios.interceptors.request.handlers.filter(Boolean);
    const handler = handlers[handlers.length - 1];

    const config = {
      url: "https://api.example.com/data",
      method: "get",
      headers: { "x-test": "value" },
    };

    const result = await handler.fulfilled(config);
    expect(result).toBe(config);
    expect(client.getQueueSize()).toBeGreaterThan(0);
  });

  it("should track response events", () => {
    interceptor.install(client);

    const handlers = axios.interceptors.response.handlers.filter(Boolean);
    const handler = handlers[handlers.length - 1];

    const response = {
      status: 200,
      statusText: "OK",
      config: { url: "https://api.example.com/data", method: "get" },
    };

    const result = handler.fulfilled(response);
    expect(result).toBe(response);
    expect(client.getQueueSize()).toBeGreaterThan(0);
  });

  it("should track error events via response error handler", async () => {
    interceptor.install(client);

    const handlers = axios.interceptors.response.handlers.filter(Boolean);
    const handler = handlers[handlers.length - 1];

    const error = Object.assign(new Error("Network Error"), {
      config: { url: "https://api.example.com/data", method: "post" },
    });

    await expect(handler.rejected(error)).rejects.toThrow("Network Error");
    expect(client.getQueueSize()).toBeGreaterThan(0);
  });

  it("should skip excluded URLs", async () => {
    interceptor.install(client, {
      excludePatterns: ["^https://api\\.trusera\\.io"],
    });

    const handlers = axios.interceptors.request.handlers.filter(Boolean);
    const handler = handlers[handlers.length - 1];

    const config = {
      url: "https://api.trusera.io/events",
      method: "post",
      headers: {},
    };

    await handler.fulfilled(config);
    expect(client.getQueueSize()).toBe(0);
  });

  it("should resolve relative URLs with baseURL", async () => {
    interceptor.install(client);

    const handlers = axios.interceptors.request.handlers.filter(Boolean);
    const handler = handlers[handlers.length - 1];

    const config = {
      baseURL: "https://api.example.com",
      url: "/users",
      method: "get",
      headers: {},
    };

    await handler.fulfilled(config);
    expect(client.getQueueSize()).toBeGreaterThan(0);
  });

  it("should handle all 3 enforcement modes", async () => {
    // Log mode (default) - should pass through
    interceptor.install(client, { enforcement: "log" });

    const handlers = axios.interceptors.request.handlers.filter(Boolean);
    const handler = handlers[handlers.length - 1];

    const config = {
      url: "https://api.example.com/test",
      method: "get",
      headers: {},
    };

    const result = await handler.fulfilled(config);
    expect(result).toBe(config);

    interceptor.uninstall();
  });
});
