/**
 * Standalone interceptor for Trusera SDK.
 *
 * Works WITHOUT any API key or platform connection.
 * Evaluates requests against local Cedar policies and logs events to disk.
 *
 * @example
 * ```typescript
 * import { StandaloneInterceptor } from "trusera-sdk";
 *
 * const interceptor = new StandaloneInterceptor({
 *   policyFile: ".cedar/ai-policy.cedar",
 *   enforcement: "block",
 *   logFile: "agent-events.jsonl",
 *   excludePatterns: ["api\\.trusera\\."],
 *   debug: false
 * });
 *
 * interceptor.install();
 *
 * // All fetch calls are now monitored
 * await fetch("https://api.github.com/repos/test");
 *
 * interceptor.uninstall();
 * ```
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { CedarEvaluator } from "./cedar.js";
import type { PolicyDecision } from "./cedar.js";

const MAX_PATTERN_LENGTH = 500;
const EVIL_REGEX_PATTERNS = /(\.\*){2,}|(\w\+){2,}|\(\[^[^\]]*\]\*\)\*|\(\.\+\)\+/;

function safeRegExp(pattern: string): RegExp | null {
  if (pattern.length > MAX_PATTERN_LENGTH) {
    console.warn(`[Trusera Standalone] Exclude pattern too long, skipping`);
    return null;
  }
  if (EVIL_REGEX_PATTERNS.test(pattern)) {
    console.warn(`[Trusera Standalone] Potentially dangerous regex pattern, skipping`);
    return null;
  }
  try {
    return new RegExp(pattern);
  } catch {
    console.warn(`[Trusera Standalone] Invalid regex pattern, skipping`);
    return null;
  }
}

/**
 * Enforcement modes for policy violations.
 */
export type StandaloneEnforcementMode = "block" | "warn" | "log";

/**
 * Configuration for the standalone interceptor.
 */
export interface StandaloneInterceptorOptions {
  /** Path to Cedar policy file (optional) */
  policyFile?: string;
  /** How to handle policy violations (default: "log") */
  enforcement?: StandaloneEnforcementMode;
  /** Path to JSONL log file for events (optional) */
  logFile?: string;
  /** URL patterns to exclude from interception (regex strings) */
  excludePatterns?: string[];
  /** Enable debug logging to console */
  debug?: boolean;
}

/**
 * Event logged to JSONL file.
 */
interface LoggedEvent {
  timestamp: string;
  method: string;
  url: string;
  status?: number;
  duration_ms?: number;
  policy_decision: "Allow" | "Deny";
  policy_reasons?: string[];
  enforcement_action: "allowed" | "blocked" | "warned";
  error?: string;
}

/**
 * Global singleton state for standalone interceptor.
 */
let activeStandaloneInterceptor: StandaloneInterceptor | null = null;
let originalFetch: typeof globalThis.fetch | null = null;

/**
 * Standalone HTTP interceptor that works without Trusera platform.
 *
 * Evaluates requests against local Cedar policies and logs events to disk.
 * Suitable for local development, testing, and air-gapped environments.
 *
 * Only one interceptor can be active at a time.
 */
export class StandaloneInterceptor {
  private options: Required<StandaloneInterceptorOptions>;
  private excludeRegexes: RegExp[] = [];
  private isInstalled = false;
  private policyEvaluator: CedarEvaluator | null = null;
  private logFileHandle: number | null = null;

  constructor(options: StandaloneInterceptorOptions = {}) {
    this.options = {
      policyFile: options.policyFile ?? "",
      enforcement: options.enforcement ?? "log",
      logFile: options.logFile ?? "",
      excludePatterns: options.excludePatterns ?? [],
      debug: options.debug ?? false,
    };

    // Compile exclude patterns (with ReDoS protection)
    this.excludeRegexes = this.options.excludePatterns
      .map((pattern) => safeRegExp(pattern))
      .filter((r): r is RegExp => r !== null);
  }

  /**
   * Installs the standalone interceptor.
   * Loads Cedar policy (if configured) and monkey-patches globalThis.fetch.
   *
   * @throws Error if another interceptor is already installed
   */
  install(): void {
    if (activeStandaloneInterceptor !== null && activeStandaloneInterceptor !== this) {
      throw new Error(
        "Another StandaloneInterceptor is already installed. Call uninstall() first."
      );
    }

    if (this.isInstalled) {
      this.log("Interceptor already installed");
      return;
    }

    // Load Cedar policy if configured
    if (this.options.policyFile) {
      this.loadPolicy(this.options.policyFile);
    }

    // Open log file if configured
    if (this.options.logFile) {
      this.openLogFile(this.options.logFile);
    }

    // Save original fetch and install interceptor
    if (originalFetch === null) {
      originalFetch = globalThis.fetch;
    }

    globalThis.fetch = this.createInterceptedFetch();
    activeStandaloneInterceptor = this;
    this.isInstalled = true;

    this.log("Standalone interceptor installed", {
      policyFile: this.options.policyFile,
      policyRules: this.policyEvaluator?.getRuleCount() ?? 0,
      enforcement: this.options.enforcement,
      logFile: this.options.logFile,
      excludePatterns: this.options.excludePatterns.length,
    });
  }

  /**
   * Uninstalls the interceptor and restores original fetch.
   * Closes log file if open.
   */
  uninstall(): void {
    if (!this.isInstalled) {
      return;
    }

    if (originalFetch !== null) {
      globalThis.fetch = originalFetch;
      originalFetch = null; // Reset for next install
    }

    // Close log file
    if (this.logFileHandle !== null) {
      try {
        fs.closeSync(this.logFileHandle);
      } catch (error) {
        console.error("[Trusera Standalone] Error closing log file:", error);
      }
      this.logFileHandle = null;
    }

    activeStandaloneInterceptor = null;
    this.isInstalled = false;

    this.log("Standalone interceptor uninstalled");
  }

  /**
   * Loads Cedar policy from file.
   */
  private loadPolicy(policyPath: string): void {
    try {
      const absolutePath = path.resolve(policyPath);
      const policyText = fs.readFileSync(absolutePath, "utf-8");
      this.policyEvaluator = new CedarEvaluator(policyText);

      this.log("Loaded Cedar policy", {
        path: absolutePath,
        rules: this.policyEvaluator.getRuleCount(),
      });
    } catch (error) {
      const err = error as Error;
      console.error(
        `[Trusera Standalone] Failed to load policy file ${policyPath}:`,
        err.message
      );
      // Continue without policy
      this.policyEvaluator = null;
    }
  }

  /**
   * Opens log file for appending.
   */
  private openLogFile(logPath: string): void {
    try {
      const absolutePath = path.resolve(logPath);

      // Ensure directory exists
      const dir = path.dirname(absolutePath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }

      // Open file for appending
      this.logFileHandle = fs.openSync(absolutePath, "a");

      this.log("Opened log file", { path: absolutePath });
    } catch (error) {
      const err = error as Error;
      console.error(
        `[Trusera Standalone] Failed to open log file ${logPath}:`,
        err.message
      );
      this.logFileHandle = null;
    }
  }

  /**
   * Creates the intercepted fetch function.
   */
  private createInterceptedFetch(): typeof globalThis.fetch {
    const self = this;

    return async function interceptedFetch(
      input: RequestInfo | URL,
      init?: RequestInit
    ): Promise<Response> {
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

      // Extract hostname and path for policy evaluation
      let hostname = "";
      let pathname = "";
      try {
        const urlObj = new URL(url);
        hostname = urlObj.hostname;
        pathname = urlObj.pathname;
      } catch {
        // Invalid URL, skip policy check
        self.log("Invalid URL, skipping policy check", { url });
        return originalFetch(input, init);
      }

      // Evaluate policy if loaded
      let policyDecision: PolicyDecision = { decision: "Allow", reasons: [] };
      if (self.policyEvaluator) {
        policyDecision = self.policyEvaluator.evaluate({
          url,
          method,
          hostname,
          path: pathname,
        });

        if (policyDecision.decision === "Deny") {
          self.log("Policy violation detected", {
            url,
            reasons: policyDecision.reasons,
          });

          // Log the violation event
          self.logEvent({
            timestamp: new Date().toISOString(),
            method,
            url,
            policy_decision: "Deny",
            policy_reasons: policyDecision.reasons,
            enforcement_action: self.options.enforcement === "block" ? "blocked" : "warned",
          });

          // Handle based on enforcement mode
          if (self.options.enforcement === "block") {
            throw new Error(
              `[Trusera] Policy violation: ${policyDecision.reasons.join(", ")}`
            );
          } else if (self.options.enforcement === "warn") {
            console.warn(
              `[Trusera] Policy violation (allowed): ${policyDecision.reasons.join(", ")}`
            );
          }
        } else {
          self.log("Policy check passed", { url });
        }
      }

      // Execute the actual request
      const startTime = Date.now();
      let response: Response;
      let error: Error | undefined;

      try {
        response = await originalFetch(input, init);
      } catch (err) {
        error = err as Error;
        self.log("Request failed", { url, error: error.message });

        // Log error event
        const errorEvent: LoggedEvent = {
          timestamp: new Date().toISOString(),
          method,
          url,
          duration_ms: Date.now() - startTime,
          policy_decision: policyDecision.decision,
          enforcement_action: "allowed",
          error: error.message,
        };
        if (policyDecision.reasons.length > 0) {
          errorEvent.policy_reasons = policyDecision.reasons;
        }
        self.logEvent(errorEvent);

        throw error;
      }

      // Log successful response
      const successEvent: LoggedEvent = {
        timestamp: new Date().toISOString(),
        method,
        url,
        status: response.status,
        duration_ms: Date.now() - startTime,
        policy_decision: policyDecision.decision,
        enforcement_action: "allowed",
      };
      if (policyDecision.reasons.length > 0) {
        successEvent.policy_reasons = policyDecision.reasons;
      }
      self.logEvent(successEvent);

      return response;
    };
  }

  /**
   * Checks if a URL should be excluded from interception.
   */
  private shouldExclude(url: string): boolean {
    return this.excludeRegexes.some((regex) => regex.test(url));
  }

  /**
   * Logs an event to the JSONL file.
   */
  private logEvent(event: LoggedEvent): void {
    if (this.logFileHandle === null) {
      return;
    }

    try {
      const line = JSON.stringify(event) + "\n";
      fs.writeSync(this.logFileHandle, line);
    } catch (error) {
      const err = error as Error;
      console.error("[Trusera Standalone] Error writing to log file:", err.message);
    }
  }

  /**
   * Debug logging.
   */
  private log(message: string, data?: Record<string, unknown>): void {
    if (this.options.debug) {
      console.log(`[Trusera Standalone] ${message}`, data ?? "");
    }
  }
}
