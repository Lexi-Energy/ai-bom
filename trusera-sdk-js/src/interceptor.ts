import type { TruseraClient } from "./client.js";
import { EventType, createEvent } from "./events.js";

/**
 * Enforcement modes for policy violations.
 */
export type EnforcementMode = "block" | "warn" | "log";

/**
 * Configuration for the HTTP interceptor.
 */
export interface InterceptorOptions {
  /** How to handle policy violations (default: "log") */
  enforcement?: EnforcementMode;
  /** Cedar policy service URL for runtime policy checks */
  policyUrl?: string;
  /** URL patterns to exclude from interception (regex strings) */
  excludePatterns?: string[];
  /** Enable debug logging */
  debug?: boolean;
  /** When true and enforcement is "block", deny requests on policy evaluation errors (default: false) */
  failClosed?: boolean;
}

/**
 * Policy evaluation request sent to Cedar policy service.
 */
interface PolicyEvaluationRequest {
  principal: string;
  action: string;
  resource: string;
  context: Record<string, unknown>;
}

/**
 * Policy evaluation response from Cedar service.
 */
interface PolicyEvaluationResponse {
  decision: "Allow" | "Deny";
  reasons?: string[];
}

/**
 * Global interceptor singleton state.
 * Only one interceptor can be active at a time.
 */
let activeInterceptor: TruseraInterceptor | null = null;
let originalFetch: typeof globalThis.fetch | null = null;

/**
 * Safely require an optional dependency.
 * Returns null if the module is not installed.
 */
// eslint-disable-next-line @typescript-eslint/no-require-imports
function tryRequire(moduleName: string): any { // eslint-disable-line @typescript-eslint/no-explicit-any
  try {
    return require(moduleName);
  } catch {
    return null;
  }
}

const SENSITIVE_HEADERS = new Set([
  "authorization",
  "cookie",
  "set-cookie",
  "x-api-key",
  "x-n8n-api-key",
  "proxy-authorization",
]);

function redactHeaders(headers: Record<string, string>): Record<string, string> {
  const redacted: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    redacted[key] = SENSITIVE_HEADERS.has(key.toLowerCase()) ? "[REDACTED]" : value;
  }
  return redacted;
}

/** Maximum allowed regex pattern length */
const MAX_PATTERN_LENGTH = 500;

/** Patterns known to cause catastrophic backtracking */
const EVIL_REGEX_PATTERNS = /(\.\*){2,}|(\w\+){2,}|\(\[^[^\]]*\]\*\)\*|\(\.\+\)\+/;

function safeRegExp(pattern: string): RegExp | null {
  if (pattern.length > MAX_PATTERN_LENGTH) {
    console.warn(`[Trusera] Exclude pattern too long (${pattern.length} > ${MAX_PATTERN_LENGTH}), skipping`);
    return null;
  }
  if (EVIL_REGEX_PATTERNS.test(pattern)) {
    console.warn(`[Trusera] Potentially dangerous regex pattern detected, skipping: ${pattern.slice(0, 50)}`);
    return null;
  }
  try {
    return new RegExp(pattern);
  } catch (e) {
    console.warn(`[Trusera] Invalid regex pattern, skipping: ${pattern.slice(0, 50)}`);
    return null;
  }
}

/**
 * HTTP interceptor for AI agent outbound traffic.
 * Monkey-patches globalThis.fetch, axios, and undici to intercept all HTTP calls,
 * evaluate them against Cedar policies, and track events.
 *
 * @example
 * ```typescript
 * const client = new TruseraClient({ apiKey: "tsk_xxx" });
 * const interceptor = new TruseraInterceptor();
 *
 * interceptor.install(client, {
 *   enforcement: "block",
 *   policyUrl: "https://policy.trusera.io/evaluate",
 *   excludePatterns: ["^https://api\\.trusera\\.io/.*"]
 * });
 *
 * // All fetch/axios/undici calls are now intercepted
 * await fetch("https://api.github.com/repos/test"); // Tracked + policy-checked
 *
 * interceptor.uninstall(); // Restore originals
 * ```
 */
export class TruseraInterceptor {
  private client: TruseraClient | null = null;
  private options: Required<InterceptorOptions> = {
    enforcement: "log",
    policyUrl: "",
    excludePatterns: [],
    debug: false,
    failClosed: false,
  };
  private excludeRegexes: RegExp[] = [];
  private isInstalled = false;

  // Axios interception state
  private axiosDefault: any = null; // eslint-disable-line @typescript-eslint/no-explicit-any
  private axiosRequestId: number | null = null;
  private axiosResponseId: number | null = null;

  // Undici interception state
  private undiciMod: any = null; // eslint-disable-line @typescript-eslint/no-explicit-any
  private origUndiciRequest: ((...args: any[]) => any) | null = null; // eslint-disable-line @typescript-eslint/no-explicit-any
  private origUndiciFetch: ((...args: any[]) => any) | null = null; // eslint-disable-line @typescript-eslint/no-explicit-any

  /**
   * Installs the HTTP interceptor by patching globalThis.fetch, axios, and undici.
   * Only one interceptor can be active at a time.
   *
   * @param client - TruseraClient instance for event tracking
   * @param options - Interceptor configuration
   * @throws Error if another interceptor is already installed
   */
  install(client: TruseraClient, options: InterceptorOptions = {}): void {
    if (activeInterceptor !== null && activeInterceptor !== this) {
      throw new Error("Another TruseraInterceptor is already installed. Call uninstall() first.");
    }

    if (this.isInstalled) {
      this.log("Interceptor already installed");
      return;
    }

    this.client = client;
    this.options = {
      enforcement: options.enforcement ?? "log",
      policyUrl: options.policyUrl ?? "",
      excludePatterns: options.excludePatterns ?? [],
      debug: options.debug ?? false,
      failClosed: options.failClosed ?? false,
    };

    // Compile exclude patterns to regexes (with ReDoS protection)
    this.excludeRegexes = this.options.excludePatterns
      .map((pattern) => safeRegExp(pattern))
      .filter((r): r is RegExp => r !== null);

    // Save original fetch and install interceptor
    if (originalFetch === null) {
      originalFetch = globalThis.fetch;
    }

    globalThis.fetch = this.createInterceptedFetch();
    activeInterceptor = this;
    this.isInstalled = true;

    // Install optional library interceptors (graceful skip if not available)
    this._installAxios();
    this._installUndici();

    this.log("Interceptor installed", {
      enforcement: this.options.enforcement,
      policyUrl: this.options.policyUrl,
      excludePatterns: this.options.excludePatterns.length,
    });
  }

  /**
   * Uninstalls the interceptor and restores original fetch, axios, and undici.
   */
  uninstall(): void {
    if (!this.isInstalled) {
      return;
    }

    if (originalFetch !== null) {
      globalThis.fetch = originalFetch;
      originalFetch = null; // Reset for next install
    }

    this._uninstallAxios();
    this._uninstallUndici();

    activeInterceptor = null;
    this.isInstalled = false;
    this.client = null;

    this.log("Interceptor uninstalled");
  }

  // ─── Shared evaluation + tracking ───────────────────────────────────

  /**
   * Shared pre-request evaluation: checks excludes, evaluates Cedar policy,
   * tracks API_CALL event, and returns whether the request should be blocked.
   * Used by fetch, axios, and undici interceptors.
   */
  private async _evaluateAndTrack(
    url: string,
    method: string,
    headers: Record<string, string>
  ): Promise<{ blocked: boolean; reasons: string[] }> {
    if (this.shouldExclude(url)) {
      return { blocked: false, reasons: [] };
    }

    const eventName = this.generateEventName(url, method);

    const event = createEvent(
      EventType.API_CALL,
      eventName,
      { method, url, headers: redactHeaders(headers) },
      { interception_mode: this.options.enforcement }
    );

    // Evaluate policy if configured
    if (this.options.policyUrl) {
      const policyDecision = await this.evaluatePolicy(url, method, { headers, body: null });

      if (policyDecision.decision === "Deny") {
        this.log("Policy violation detected", { url, reasons: policyDecision.reasons });

        const violationEvent = createEvent(
          EventType.API_CALL,
          `${eventName}.policy_violation`,
          {
            ...event.payload,
            policy_decision: "Deny",
            policy_reasons: policyDecision.reasons,
          }
        );
        this.client?.track(violationEvent);

        if (this.options.enforcement === "block") {
          return { blocked: true, reasons: policyDecision.reasons ?? [] };
        } else if (this.options.enforcement === "warn") {
          console.warn(
            `[Trusera] Policy violation (allowed): ${policyDecision.reasons?.join(", ") ?? "Request denied"}`
          );
        }
      }
    }

    this.client?.track(event);
    return { blocked: false, reasons: [] };
  }

  // ─── Axios interceptor ─────────────────────────────────────────────

  private _installAxios(): void {
    const axios = tryRequire("axios");
    if (!axios?.interceptors) return;

    this.axiosDefault = axios;
    const self = this;

    this.axiosRequestId = axios.interceptors.request.use(
      async (config: any) => { // eslint-disable-line @typescript-eslint/no-explicit-any
        const url = self._resolveAxiosUrl(config);
        const method = ((config.method as string | undefined) ?? "get").toUpperCase();
        const headers = self._flattenHeaders(config.headers);

        const { blocked, reasons } = await self._evaluateAndTrack(url, method, headers);
        if (blocked) {
          throw new Error(
            `[Trusera] Policy violation: ${reasons.join(", ") || "Request denied"}`
          );
        }
        return config;
      }
    );

    this.axiosResponseId = axios.interceptors.response.use(
      (response: any) => { // eslint-disable-line @typescript-eslint/no-explicit-any
        const url = self._resolveAxiosUrl(response.config);
        const method = ((response.config?.method as string | undefined) ?? "get").toUpperCase();

        if (!self.shouldExclude(url)) {
          const event = createEvent(
            EventType.API_CALL,
            `${self.generateEventName(url, method)}.response`,
            { method, url, status: response.status, status_text: response.statusText }
          );
          self.client?.track(event);
        }
        return response;
      },
      (error: any) => { // eslint-disable-line @typescript-eslint/no-explicit-any
        const config = error?.config;
        if (config) {
          const url = self._resolveAxiosUrl(config);
          const method = ((config.method as string | undefined) ?? "get").toUpperCase();

          if (!self.shouldExclude(url)) {
            const event = createEvent(
              EventType.API_CALL,
              `${self.generateEventName(url, method)}.error`,
              { method, url, error: (error as Error).message }
            );
            self.client?.track(event);
          }
        }
        return Promise.reject(error as Error);
      }
    );

    this.log("Axios interceptor installed");
  }

  private _uninstallAxios(): void {
    if (this.axiosDefault) {
      if (this.axiosRequestId !== null) {
        this.axiosDefault.interceptors.request.eject(this.axiosRequestId);
        this.axiosRequestId = null;
      }
      if (this.axiosResponseId !== null) {
        this.axiosDefault.interceptors.response.eject(this.axiosResponseId);
        this.axiosResponseId = null;
      }
      this.axiosDefault = null;
    }
  }

  private _resolveAxiosUrl(config: any): string { // eslint-disable-line @typescript-eslint/no-explicit-any
    if (!config) return "";
    const base: string = (config.baseURL as string | undefined) ?? "";
    const url: string = (config.url as string | undefined) ?? "";
    if (url.startsWith("http")) return url;
    return base ? `${base.replace(/\/$/, "")}/${url.replace(/^\//, "")}` : url;
  }

  private _flattenHeaders(headers: any): Record<string, string> { // eslint-disable-line @typescript-eslint/no-explicit-any
    if (!headers) return {};
    if (typeof headers.toJSON === "function") {
      return headers.toJSON() as Record<string, string>;
    }
    const result: Record<string, string> = {};
    for (const [k, v] of Object.entries(headers)) {
      if (typeof v === "string") result[k] = v;
    }
    return result;
  }

  // ─── Undici interceptor ─────────────────────────────────────────────

  private _installUndici(): void {
    const undici = tryRequire("undici");
    if (!undici) return;

    this.undiciMod = undici;
    const self = this;

    // Patch undici.request
    if (typeof undici.request === "function") {
      this.origUndiciRequest = undici.request;
      undici.request = async function interceptedUndiciRequest(
        url: string | URL,
        options?: Record<string, unknown>
      ): Promise<unknown> {
        const urlStr = typeof url === "string" ? url : url.href;
        const method = ((options?.method as string | undefined) ?? "GET").toUpperCase();
        const headers = self._flattenHeaders(options?.headers);

        const { blocked, reasons } = await self._evaluateAndTrack(urlStr, method, headers);
        if (blocked) {
          throw new Error(
            `[Trusera] Policy violation: ${reasons.join(", ") || "Request denied"}`
          );
        }

        const result: any = await self.origUndiciRequest!.call(undici, url, options); // eslint-disable-line @typescript-eslint/no-explicit-any

        if (!self.shouldExclude(urlStr)) {
          const event = createEvent(
            EventType.API_CALL,
            `${self.generateEventName(urlStr, method)}.response`,
            { method, url: urlStr, status: result.statusCode }
          );
          self.client?.track(event);
        }

        return result;
      };
    }

    // Patch undici.fetch
    if (typeof undici.fetch === "function") {
      this.origUndiciFetch = undici.fetch;
      undici.fetch = async function interceptedUndiciFetch(
        input: string | URL | Request,
        init?: RequestInit
      ): Promise<Response> {
        const url = typeof input === "string"
          ? input
          : input instanceof URL
            ? input.href
            : input.url;
        const method = (init?.method ?? "GET").toUpperCase();

        if (self.shouldExclude(url)) {
          return self.origUndiciFetch!.call(undici, input, init) as Promise<Response>;
        }

        const { blocked, reasons } = await self._evaluateAndTrack(url, method, {});
        if (blocked) {
          throw new Error(
            `[Trusera] Policy violation: ${reasons.join(", ") || "Request denied"}`
          );
        }

        const response = await self.origUndiciFetch!.call(undici, input, init) as Response;

        const event = createEvent(
          EventType.API_CALL,
          `${self.generateEventName(url, method)}.response`,
          { method, url, status: response.status }
        );
        self.client?.track(event);

        return response;
      };
    }

    this.log("Undici interceptor installed");
  }

  private _uninstallUndici(): void {
    if (this.undiciMod) {
      if (this.origUndiciRequest) {
        this.undiciMod.request = this.origUndiciRequest;
        this.origUndiciRequest = null;
      }
      if (this.origUndiciFetch) {
        this.undiciMod.fetch = this.origUndiciFetch;
        this.origUndiciFetch = null;
      }
      this.undiciMod = null;
    }
  }

  // ─── Fetch interceptor (core) ───────────────────────────────────────

  /**
   * Creates the intercepted fetch function.
   * This is the core of the interception logic.
   */
  private createInterceptedFetch(): typeof globalThis.fetch {
    const self = this;

    return async function interceptedFetch(
      input: RequestInfo | URL,
      init?: RequestInit
    ): Promise<Response> {
      // Ensure we have original fetch to fall back to
      if (originalFetch === null) {
        throw new Error("Original fetch not available");
      }

      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      const method = init?.method ?? "GET";

      // Check if URL should be excluded
      if (self.shouldExclude(url)) {
        self.log("Skipping excluded URL", { url });
        return originalFetch(input, init);
      }

      self.log("Intercepting request", { method, url });

      // Extract request details
      const requestData = await self.extractRequestData(input, init);

      // Create tracking event
      const event = createEvent(
        EventType.API_CALL,
        self.generateEventName(url, method),
        {
          method,
          url,
          headers: requestData.headers,
          body: requestData.body,
        },
        {
          interception_mode: self.options.enforcement,
        }
      );

      // Evaluate policy if configured
      if (self.options.policyUrl) {
        const policyDecision = await self.evaluatePolicy(url, method, requestData);

        if (policyDecision.decision === "Deny") {
          self.log("Policy violation detected", { url, reasons: policyDecision.reasons });

          // Track violation event
          const violationEvent = createEvent(
            EventType.API_CALL,
            `${self.generateEventName(url, method)}.policy_violation`,
            {
              ...event.payload,
              policy_decision: "Deny",
              policy_reasons: policyDecision.reasons,
            }
          );

          self.client?.track(violationEvent);

          // Handle based on enforcement mode
          if (self.options.enforcement === "block") {
            throw new Error(
              `[Trusera] Policy violation: ${policyDecision.reasons?.join(", ") ?? "Request denied"}`
            );
          } else if (self.options.enforcement === "warn") {
            console.warn(
              `[Trusera] Policy violation (allowed): ${policyDecision.reasons?.join(", ") ?? "Request denied"}`
            );
          }
        } else {
          self.log("Policy check passed", { url });
        }
      }

      // Track the API call event
      self.client?.track(event);

      // Execute the actual request
      const startTime = Date.now();
      let response: Response;
      let error: Error | undefined;

      try {
        response = await originalFetch(input, init);
      } catch (err) {
        error = err as Error;
        self.log("Request failed", { url, error: error.message });

        // Track error event
        const errorEvent = createEvent(
          EventType.API_CALL,
          `${self.generateEventName(url, method)}.error`,
          {
            ...event.payload,
            error: error.message,
            duration_ms: Date.now() - startTime,
          }
        );
        self.client?.track(errorEvent);

        throw error;
      }

      // Track response
      const responseEvent = createEvent(
        EventType.API_CALL,
        `${self.generateEventName(url, method)}.response`,
        {
          ...event.payload,
          status: response.status,
          status_text: response.statusText,
          duration_ms: Date.now() - startTime,
          response_headers: (() => { const h: Record<string, string> = {}; response.headers.forEach((v, k) => { h[k] = v; }); return redactHeaders(h); })(),
        }
      );
      self.client?.track(responseEvent);

      return response;
    };
  }

  // ─── Shared utilities ───────────────────────────────────────────────

  /**
   * Checks if a URL should be excluded from interception.
   */
  private shouldExclude(url: string): boolean {
    return this.excludeRegexes.some((regex) => regex.test(url));
  }

  /**
   * Extracts request data for policy evaluation and tracking.
   */
  private async extractRequestData(
    _input: RequestInfo | URL,
    init?: RequestInit
  ): Promise<{ headers: Record<string, string>; body: string | null }> {
    const headers: Record<string, string> = {};

    // Extract headers
    if (init?.headers) {
      if (init.headers instanceof Headers) {
        init.headers.forEach((value, key) => {
          headers[key] = value;
        });
      } else if (Array.isArray(init.headers)) {
        for (const [key, value] of init.headers) {
          headers[key] = value;
        }
      } else {
        Object.assign(headers, init.headers);
      }
    }

    // Extract body (if present and serializable)
    let body: string | null = null;
    if (init?.body) {
      if (typeof init.body === "string") {
        body = init.body;
      } else if (init.body instanceof URLSearchParams) {
        body = init.body.toString();
      } else if (init.body instanceof FormData) {
        body = "[FormData]";
      } else {
        body = "[Binary data]";
      }
    }

    return { headers: redactHeaders(headers), body };
  }

  /**
   * Evaluates request against Cedar policies.
   */
  private async evaluatePolicy(
    url: string,
    method: string,
    requestData: { headers: Record<string, string>; body: string | null }
  ): Promise<PolicyEvaluationResponse> {
    if (!this.options.policyUrl || !originalFetch) {
      return { decision: "Allow" };
    }

    try {
      const evaluationRequest: PolicyEvaluationRequest = {
        principal: this.client?.getAgentId() ?? "unknown-agent",
        action: `http.${method.toLowerCase()}`,
        resource: url,
        context: {
          headers: requestData.headers,
          body_present: requestData.body !== null,
          timestamp: new Date().toISOString(),
        },
      };

      const response = await originalFetch(this.options.policyUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(evaluationRequest),
      });

      if (!response.ok) {
        console.error(`[Trusera] Policy evaluation failed: ${response.status}`);
        if (this.options.failClosed && this.options.enforcement === "block") {
          console.warn("[Trusera] FAIL-CLOSED: Policy service returned error, denying request");
          return { decision: "Deny", reasons: [`Policy service error: ${response.status}`] };
        }
        console.warn("[Trusera] FAIL-OPEN: Policy service returned error, allowing request");
        return { decision: "Allow" }; // Fail open
      }

      return (await response.json()) as PolicyEvaluationResponse;
    } catch (error) {
      console.error("[Trusera] Policy evaluation error:", error);
      if (this.options.failClosed && this.options.enforcement === "block") {
        console.warn("[Trusera] FAIL-CLOSED: Policy evaluation failed, denying request");
        return { decision: "Deny", reasons: ["Policy evaluation failed (fail-closed mode)"] };
      }
      console.warn("[Trusera] FAIL-OPEN: Policy evaluation failed, allowing request");
      return { decision: "Allow" }; // Fail open on error
    }
  }

  /**
   * Generates a structured event name from URL and method.
   */
  private generateEventName(url: string, method: string): string {
    try {
      const urlObj = new URL(url);
      const hostname = urlObj.hostname.replace(/\./g, "_");
      const path = urlObj.pathname.split("/").filter(Boolean).join("_") || "root";
      return `http.${method.toLowerCase()}.${hostname}.${path}`;
    } catch {
      return `http.${method.toLowerCase()}.invalid_url`;
    }
  }

  private log(message: string, data?: Record<string, unknown>): void {
    if (this.options.debug) {
      console.log(`[Trusera Interceptor] ${message}`, data ?? "");
    }
  }
}
