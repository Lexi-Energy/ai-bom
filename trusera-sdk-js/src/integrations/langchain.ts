import type { TruseraClient } from "../client.js";
import { EventType, createEvent, type Event } from "../events.js";
import type { EnforcementMode } from "../interceptor.js";
import type { CedarEvaluator } from "../cedar.js";

/**
 * Options for configuring the LangChain handler with policy enforcement.
 */
export interface LangChainHandlerOptions {
  /** Enforcement mode for policy violations (default: none/passive tracking only) */
  enforcement?: EnforcementMode;
  /** Cedar policy evaluator for local policy checks */
  cedarEvaluator?: CedarEvaluator;
}

/**
 * LangChain.js callback handler for Trusera integration.
 * Implements the BaseCallbackHandler interface to track LLM calls,
 * tool executions, and chain steps.
 *
 * Supports optional Cedar policy enforcement: block throws, warn logs, log is silent.
 * Without options, operates in passive tracking mode (full backward compatibility).
 *
 * Note: This uses structural typing to avoid requiring langchain as a dependency.
 * If langchain types change, this will need updating.
 *
 * @example
 * ```typescript
 * import { ChatOpenAI } from "langchain/chat_models/openai";
 * import { TruseraClient, TruseraLangChainHandler, CedarEvaluator } from "trusera-sdk";
 *
 * const client = new TruseraClient({ apiKey: "tsk_xxx" });
 *
 * // Passive tracking (no enforcement)
 * const handler = new TruseraLangChainHandler(client);
 *
 * // With Cedar policy enforcement
 * const evaluator = new CedarEvaluator(policyText);
 * const enforcedHandler = new TruseraLangChainHandler(client, {
 *   enforcement: "block",
 *   cedarEvaluator: evaluator
 * });
 *
 * const model = new ChatOpenAI({ callbacks: [enforcedHandler] });
 * await model.invoke("What is AI safety?");
 *
 * await client.close();
 * ```
 */
export class TruseraLangChainHandler {
  private readonly client: TruseraClient;
  private readonly pendingEvents = new Map<string, Event>();
  private readonly enforcement: EnforcementMode | undefined;
  private readonly cedarEvaluator: CedarEvaluator | undefined;

  constructor(client: TruseraClient, options?: LangChainHandlerOptions) {
    this.client = client;
    this.enforcement = options?.enforcement;
    this.cedarEvaluator = options?.cedarEvaluator;
  }

  /**
   * Called when an LLM starts running.
   */
  handleLLMStart(
    llm: { name: string },
    prompts: string[],
    runId: string,
    parentRunId?: string,
    extraParams?: Record<string, unknown>,
    tags?: string[],
    metadata?: Record<string, unknown>
  ): void {
    // Evaluate policy if enforcement is configured
    if (this.enforcement && this.cedarEvaluator) {
      const decision = this.cedarEvaluator.evaluate({
        url: `langchain://llm/${llm.name}`,
        method: "INVOKE",
        hostname: "langchain",
        path: `/llm/${llm.name}`,
      });

      if (decision.decision === "Deny") {
        this._handleViolation(
          `langchain.llm.${llm.name}`,
          EventType.LLM_INVOKE,
          decision.reasons,
          runId
        );

        if (this.enforcement === "block") {
          throw new Error(
            `[Trusera] Policy violation: LLM ${llm.name} denied - ${decision.reasons.join(", ")}`
          );
        }
      }
    }

    const event = createEvent(
      EventType.LLM_INVOKE,
      `langchain.llm.${llm.name}`,
      {
        prompts,
        prompt_count: prompts.length,
        invocation_params: extraParams ?? {},
      },
      {
        run_id: runId,
        parent_run_id: parentRunId,
        tags: tags ?? [],
        ...metadata,
      }
    );

    this.pendingEvents.set(runId, event);
    this.client.track(event);
  }

  /**
   * Called when an LLM finishes running.
   */
  handleLLMEnd(
    output: { generations: Array<Array<{ text: string }>> },
    runId: string
  ): void {
    const startEvent = this.pendingEvents.get(runId);
    if (!startEvent) return;

    const texts = output.generations.flatMap((gen) => gen.map((g) => g.text));

    const event = createEvent(
      EventType.LLM_INVOKE,
      `${startEvent.name}.completed`,
      {
        ...startEvent.payload,
        outputs: texts,
        output_count: texts.length,
      },
      startEvent.metadata
    );

    this.client.track(event);
    this.pendingEvents.delete(runId);
  }

  /**
   * Called when an LLM encounters an error.
   */
  handleLLMError(
    error: Error,
    runId: string
  ): void {
    const startEvent = this.pendingEvents.get(runId);
    if (!startEvent) return;

    const event = createEvent(
      EventType.LLM_INVOKE,
      `${startEvent.name}.error`,
      {
        ...startEvent.payload,
        error: error.message,
        error_type: error.name,
      },
      startEvent.metadata
    );

    this.client.track(event);
    this.pendingEvents.delete(runId);
  }

  /**
   * Called when a tool starts running.
   */
  handleToolStart(
    tool: { name: string },
    input: string,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>
  ): void {
    // Evaluate policy if enforcement is configured
    if (this.enforcement && this.cedarEvaluator) {
      const decision = this.cedarEvaluator.evaluate({
        url: `langchain://tool/${tool.name}`,
        method: "EXECUTE",
        hostname: "langchain",
        path: `/tool/${tool.name}`,
      });

      if (decision.decision === "Deny") {
        this._handleViolation(
          `langchain.tool.${tool.name}`,
          EventType.TOOL_CALL,
          decision.reasons,
          runId
        );

        if (this.enforcement === "block") {
          throw new Error(
            `[Trusera] Policy violation: tool ${tool.name} denied - ${decision.reasons.join(", ")}`
          );
        }
      }
    }

    const event = createEvent(
      EventType.TOOL_CALL,
      `langchain.tool.${tool.name}`,
      {
        input,
        input_length: input.length,
      },
      {
        run_id: runId,
        parent_run_id: parentRunId,
        tags: tags ?? [],
        ...metadata,
      }
    );

    this.pendingEvents.set(runId, event);
    this.client.track(event);
  }

  /**
   * Called when a tool finishes running.
   */
  handleToolEnd(
    output: string,
    runId: string
  ): void {
    const startEvent = this.pendingEvents.get(runId);
    if (!startEvent) return;

    const event = createEvent(
      EventType.TOOL_CALL,
      `${startEvent.name}.completed`,
      {
        ...startEvent.payload,
        output,
        output_length: output.length,
      },
      startEvent.metadata
    );

    this.client.track(event);
    this.pendingEvents.delete(runId);
  }

  /**
   * Called when a tool encounters an error.
   */
  handleToolError(
    error: Error,
    runId: string
  ): void {
    const startEvent = this.pendingEvents.get(runId);
    if (!startEvent) return;

    const event = createEvent(
      EventType.TOOL_CALL,
      `${startEvent.name}.error`,
      {
        ...startEvent.payload,
        error: error.message,
        error_type: error.name,
      },
      startEvent.metadata
    );

    this.client.track(event);
    this.pendingEvents.delete(runId);
  }

  /**
   * Called when a chain starts running.
   */
  handleChainStart(
    chain: { name: string },
    inputs: Record<string, unknown>,
    runId: string,
    parentRunId?: string,
    tags?: string[],
    metadata?: Record<string, unknown>
  ): void {
    const event = createEvent(
      EventType.DECISION,
      `langchain.chain.${chain.name}`,
      {
        inputs,
        input_keys: Object.keys(inputs),
      },
      {
        run_id: runId,
        parent_run_id: parentRunId,
        tags: tags ?? [],
        ...metadata,
      }
    );

    this.pendingEvents.set(runId, event);
    this.client.track(event);
  }

  /**
   * Called when a chain finishes running.
   */
  handleChainEnd(
    outputs: Record<string, unknown>,
    runId: string
  ): void {
    const startEvent = this.pendingEvents.get(runId);
    if (!startEvent) return;

    const event = createEvent(
      EventType.DECISION,
      `${startEvent.name}.completed`,
      {
        ...startEvent.payload,
        outputs,
        output_keys: Object.keys(outputs),
      },
      startEvent.metadata
    );

    this.client.track(event);
    this.pendingEvents.delete(runId);
  }

  /**
   * Called when a chain encounters an error.
   */
  handleChainError(
    error: Error,
    runId: string
  ): void {
    const startEvent = this.pendingEvents.get(runId);
    if (!startEvent) return;

    const event = createEvent(
      EventType.DECISION,
      `${startEvent.name}.error`,
      {
        ...startEvent.payload,
        error: error.message,
        error_type: error.name,
      },
      startEvent.metadata
    );

    this.client.track(event);
    this.pendingEvents.delete(runId);
  }

  /**
   * Returns the number of pending (incomplete) events.
   * Useful for testing and debugging.
   */
  getPendingEventCount(): number {
    return this.pendingEvents.size;
  }

  /**
   * Clears all pending events.
   * Should only be used in testing or error recovery.
   */
  clearPendingEvents(): void {
    this.pendingEvents.clear();
  }

  /**
   * Handle a policy violation: track POLICY_VIOLATION event and warn if needed.
   */
  private _handleViolation(
    name: string,
    eventType: EventType,
    reasons: string[],
    runId: string
  ): void {
    const violationEvent = createEvent(
      eventType,
      `${name}.policy_violation`,
      {
        policy_decision: "Deny",
        policy_reasons: reasons,
      },
      { run_id: runId }
    );
    this.client.track(violationEvent);

    if (this.enforcement === "warn") {
      console.warn(
        `[Trusera] Policy violation (allowed): ${reasons.join(", ")}`
      );
    }
  }
}
