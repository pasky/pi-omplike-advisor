/**
 * /advisor — a persistent second model that reviews the main agent's work each
 * turn and injects concise advice inline. Port of oh-my-pi's advisor onto
 * upstream pi's extension API.
 *
 * Enable with `/advisor on` (persisted). The advisor model defaults to
 * openrouter/z-ai/glm-5.2 (override via an "advisor" entry in modes.json).
 *
 * Delivery model. Nothing here is a hard interrupt: upstream pi's extension
 * surface delivers via `steer` (the message folds in at the agent's next step
 * boundary; `triggerTurn` additionally wakes an idle agent). We never call
 * `abort()`. So:
 *
 *   nit      → delivered immediately (steer + triggerTurn), tagged as raised
 *              about an earlier step. Low-stakes; mild staleness is fine.
 *   concern  → ALWAYS held, never steered on first emission.
 *   blocker  → ALWAYS held, never steered on first emission.
 *
 * Why always-hold for high severity: the advisor reviews turn N asynchronously
 * (seconds), so by the time any advice could land the primary has almost always
 * done follow-up work — the advice is stale. Instead we hold it and let the next
 * review reconfirm it (held notes ride a reconfirm preamble; the advisor re-
 * raises survivors, stays silent on the resolved ones).
 *
 * Catch-up block: while a high-severity note is held — or whenever a turn is
 * about to idle — we stall the primary's next step (by awaiting in the `turn_end`
 * hook, which the agent loop awaits) so the advisor can catch up. The wait backs
 * off 15s→30s→60s… capped at 120s, is Escape-abortable, and shows a notice. Once
 * the advisor settles, surviving held notes are steered in against the now-unraced
 * state. This is a deliberate throttle (omp's syncBacklog idea).
 *
 * An optional WATCHDOG.md in the cwd is appended to the advisor's system prompt
 * (advisor-only guidance: review priorities, project traps).
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import { Agent, type AgentToolResult } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Model, ToolResultMessage, UserMessage } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { convertToLlm, createReadOnlyTools } from "@earendil-works/pi-coding-agent";
import { Container, Text } from "@earendil-works/pi-tui";
import { Type } from "typebox";

import { resolveModelAndThinking } from "./lib/mode-utils.js";

// ===========================================================================
// Advisor core — persistent second model that watches the main agent.
//
// Port of oh-my-pi's advisor onto upstream pi's public extension surface. The
// advisor is a long-lived `Agent` with its own model + read-only tools
// (read/grep/find) and one `advise` tool. It is fed the primary transcript one
// turn-delta at a time and may inject concise advice back. It is NOT an
// executor: it cannot edit, run commands, or change session state.
// ===========================================================================

export type AdvisorSeverity = "nit" | "concern" | "blocker";
export interface AdvisorNote {
	note: string;
	severity?: AdvisorSeverity;
}

// ---- advise tool (agent-core tool; lives only on the advisor agent) ----

const adviseSchema = Type.Object({
	note: Type.String({
		description: "One concrete piece of advice for the agent you are watching. Terse, specific, actionable.",
	}),
	severity: Type.Optional(
		Type.Union([Type.Literal("nit"), Type.Literal("concern"), Type.Literal("blocker")], {
			description: "How strongly to weigh this. Omit for a plain nit.",
		}),
	),
});

const SEVERITY_RANK: Record<AdvisorSeverity, number> = { nit: 1, concern: 2, blocker: 3 };
const rankOf = (s: AdvisorSeverity | undefined): number => SEVERITY_RANK[s ?? "nit"];
const dedupeKey = (note: string): string => note.trim().replace(/\s+/g, " ");
/** High severity (concern/blocker) is always held + reconfirmed; nits deliver now. */
export const isHighSeverity = (s: AdvisorSeverity | undefined): boolean => s === "concern" || s === "blocker";

/** Catch-up block backoff: base, 2×, 4×… capped. consecutive=0 → base (15s default). */
export function nextBackoffMs(consecutive: number, baseMs = 15_000, capMs = 120_000): number {
	return Math.min(capMs, baseMs * 2 ** Math.max(0, consecutive));
}

/**
 * A turn is terminal (the agent is about to go idle) when its assistant message
 * issued no tool calls — the agent-loop's inner loop exits unless something is
 * steered in. We block-until-settled on terminal turns so a blocker the advisor
 * raises about the final turn is caught before control returns to the user.
 *
 * Approximation: a turn WITH tool calls can still end the run if a tool returns
 * `terminate` or a stop hook fires; we'd classify that non-terminal. The cost is
 * only a *delay*, not a loss — a held note still rides the next turn's catch-up
 * block; the sole gap is a brand-new blocker raised about such a turn (nothing
 * previously held), which then lands on the next user turn instead of before idle.
 */
export function isTerminalTurn(message: { content?: ReadonlyArray<{ type: string }> } | undefined): boolean {
	return !(message?.content ?? []).some((c) => c.type === "toolCall");
}

/** Structural slice of AdvisorRuntime the catch-up block needs (so it's testable). */
export interface TurnBlockRuntime {
	readonly hasHeld: boolean;
	takeHeld(): AdvisorNote[];
	waitUntilSettled(timeoutMs: number, signal?: AbortSignal): Promise<"settled" | "timeout" | "aborted" | "failed">;
}

/**
 * The catch-up block, run once per primary `turn_end` (after the delta is pushed).
 * Returns the next `consecutiveBlocks` count for the caller to carry.
 *
 * - Non-terminal turn with nothing held → no block (streak resets to 0).
 * - Otherwise block, racing advisor-settled vs a timeout vs the abort signal:
 *     - terminal → timeout = cap (block until the advisor finishes the last turn).
 *     - mid-run  → timeout = backoff(consecutiveBlocks); on timeout, keep the held
 *                  notes and lengthen the next wait (return consecutiveBlocks+1).
 * - On settle: steer in whatever survived reconfirmation (may be empty), reset streak.
 * - On timeout / failed reconfirm (advisor errored out): non-terminal keeps the
 *   held notes and lengthens the next wait; terminal delivers best-effort (the
 *   agent did no follow-up — it stopped — so held notes are current, and it's the
 *   last chance before control returns to the user).
 * - On abort (user hit Escape): bail, keep held notes + streak.
 */
export async function runTurnBlock(opts: {
	terminal: boolean;
	runtime: TurnBlockRuntime;
	consecutiveBlocks: number;
	baseMs?: number;
	capMs?: number;
	signal?: AbortSignal;
	notify: (msg: string) => void;
	deliverHeld: (notes: AdvisorNote[], opts?: { terminal?: boolean }) => void;
}): Promise<number> {
	const { terminal, runtime } = opts;
	const baseMs = opts.baseMs ?? 15_000;
	const capMs = opts.capMs ?? 120_000;
	if (!terminal && !runtime.hasHeld) return 0;

	const timeoutMs = terminal ? capMs : nextBackoffMs(opts.consecutiveBlocks, baseMs, capMs);
	opts.notify(
		terminal
			? "advisor: catching up before the turn ends…"
			: `advisor: waiting up to ${Math.round(timeoutMs / 1000)}s to catch up…`,
	);

	const result = await runtime.waitUntilSettled(timeoutMs, opts.signal);
	if (result === "aborted") return opts.consecutiveBlocks; // user bailed; keep held + streak
	if (result === "settled") {
		// Only a successful reconfirmation settles; the advisor has pruned recanted
		// notes, so #held is the confirmed survivor set.
		const held = runtime.takeHeld();
		if (held.length) opts.deliverHeld(held, { terminal });
		return 0;
	}
	// timeout OR failed (advisor errored 3x and dropped the reconfirm). Either way
	// the held notes are NOT confirmed.
	if (terminal) {
		const held = runtime.takeHeld();
		if (held.length) {
			opts.deliverHeld(held, { terminal: true });
			opts.notify("advisor didn't reconfirm in time; delivering held advice anyway");
		}
		return 0;
	}
	return opts.consecutiveBlocks + 1; // mid-run: keep held unconfirmed, lengthen next wait
}

/**
 * Render held advisories as a reconfirm preamble prepended to the next review.
 * Empty string when nothing is held.
 */
export function formatReconfirmPreamble(held: readonly AdvisorNote[]): string {
	if (!held.length) return "";
	const items = held.map((n) => `- [${(n.severity ?? "nit").toUpperCase()}] ${n.note}`).join("\n");
	return [
		"### Held advisories — reconfirm",
		"",
		"You raised these on an earlier step; they were held pending reconfirmation, because by now the agent may have already addressed them. Re-check each against the latest activity below.",
		"For every item that STILL applies, call `advise` again — same severity, or higher if it's gotten worse; never lower it. Say nothing for the rest — silence drops them. Do NOT call `advise` to announce that an item is resolved or that all are cleared; just stay silent.",
		"",
		items,
		"",
		"---",
		"",
	].join("\n");
}

/** Parse the hidden `/advisor test <nit|concern|blocker> <note>` test hook args. */
export function parseAdvisorTestArgs(args: string): { severity: AdvisorSeverity; note: string } | null {
	const m = args.trim().match(/^test\s+(nit|concern|blocker)\s+([\s\S]+)$/i);
	if (!m) return null;
	return { severity: m[1].toLowerCase() as AdvisorSeverity, note: m[2].trim() };
}

/**
 * The advise tool. Dedupes by normalized note text + severity rank: a repeat at
 * the same-or-lower severity is dropped, a real escalation (nit→concern→blocker)
 * passes through. Dedup is recorded only when the note is actually *delivered*
 * (`onAdvice` returns true) — a note that is held for reconfirmation returns
 * false and is left unrecorded so it can re-fire and land once it's confirmed.
 */
export class AdviseTool {
	readonly name = "advise";
	readonly label = "Advise";
	readonly description =
		"Send one concrete, ACTIONABLE piece of advice to the agent you are watching. Use sparingly; stay silent when nothing matters. Call it to head off likely-wrong or materially wasteful work. NEVER call it to report status, acknowledge, confirm, summarize, or signal that all is well / resolved / nothing-further-needed — in those cases emit nothing.";
	readonly parameters = adviseSchema as any;
	#delivered = new Map<string, number>();

	// onAdvice returns true if the note was delivered, false if it was held
	// (high severity) and should be re-offered/reconfirmed later.
	constructor(private readonly onAdvice: (note: string, severity?: AdvisorSeverity) => boolean) {}

	resetDelivered(): void {
		this.#delivered.clear();
	}

	/**
	 * Record a note as delivered so a later same-or-lower-severity repeat is
	 * deduped. Called by the catch-up block when it steers a held note in (held
	 * notes go through `onAdvice`→false, which intentionally does NOT record, so
	 * the actual delivery point must).
	 */
	markDelivered(note: string, severity?: AdvisorSeverity): void {
		this.#delivered.set(dedupeKey(note), rankOf(severity));
	}

	async execute(_id: string, args: { note: string; severity?: AdvisorSeverity }): Promise<AgentToolResult<unknown>> {
		const key = dedupeKey(args.note);
		const rank = rankOf(args.severity);
		const prev = this.#delivered.get(key) ?? 0;
		if (rank <= prev) {
			return { content: [{ type: "text", text: "Duplicate advice ignored." }], details: { ...args, dropped: true } };
		}
		const delivered = this.onAdvice(args.note, args.severity);
		if (!delivered) {
			// Held for reconfirmation: leave undealt so it can re-fire and land later.
			return { content: [{ type: "text", text: "Held pending reconfirmation." }], details: { ...args, held: true } };
		}
		this.#delivered.set(key, rank);
		return { content: [{ type: "text", text: "Recorded." }], details: { ...args } };
	}
}

// ---- advisory rendering for the primary transcript ----

const ADVISOR_GUIDANCE = "weigh, don't blindly obey";
const escapeXml = (s: string): string => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/**
 * Render notes as the agent-facing message body: one `<advisory>` per note.
 * `stale` adds a `context` attribute noting the advice is about an earlier step
 * (used for nits, which the advisor always raises a little behind the agent).
 * `finalAnswer` appends guidance for advice that arrives AFTER the agent had
 * already returned a final answer this turn (a terminal/stop turn): if the agent
 * acts on it, it should reply with a fresh, self-contained final answer rather
 * than a terse follow-up — so the user reads one complete answer, not a
 * back-and-forth thread it has to stitch together.
 */
export function formatAdvisoryContent(notes: readonly AdvisorNote[], opts?: { stale?: boolean; finalAnswer?: boolean }): string {
	const context = opts?.stale ? ` context="raised about an earlier step"` : "";
	const body = notes
		.map((n) => {
			const sev = n.severity ? ` severity="${n.severity}"` : "";
			return `<advisory${sev}${context} guidance="${ADVISOR_GUIDANCE}">\n${escapeXml(n.note)}\n</advisory>`;
		})
		.join("\n");
	if (!opts?.finalAnswer) return body;
	return `${body}\n\nYou had already returned a final answer to the user this turn. If you act on the advice above, respond with a new, self-contained final answer that fully stands on its own — do NOT write a terse follow-up that assumes the user read your previous message. The user should be able to read your new reply alone and get the complete answer.`;
}

// ---- transcript delta formatting (primary turn → markdown for the advisor) ----

// No truncation of the delta. The advisor is a peer reviewer (its own model, its
// own read/grep/find), not a cheap/lightweight pass — nothing in the design says
// otherwise. It must see what the main model saw, verbatim; clipping fields just
// hid the part it needed to verify and bred false "didn't persist"/"garbled"
// advice. (The advisor CAN re-read to verify — system prompt — but that's about
// its actions, not a license to starve its input.)
//
// Input-budget policy (advisor self-compaction): the advisor's context is a pure
// linear accumulation of INDEPENDENT turn deltas — no essential cross-turn state
// lives in the agent's message history (held notes live in #held and ride the
// reconfirm preamble, not the transcript). So when the advisor's own context
// approaches the window it self-compacts: #drain clears ONLY the agent's message
// history (#softReset) and replays the current batch into a fresh context. Two
// triggers — PROACTIVE (before prompting, when usage crosses COMPACT_AT_PERCENT)
// and REACTIVE (a review that still comes back stopReason=="length"). The reactive
// path is loop-safe: if the agent was ALREADY fresh and still overflowed, the
// single batch genuinely doesn't fit, so we stop self-compacting and fall through
// to the normal failed-review handling instead of spinning. This replaces the old
// behavior (overflow -> fail review -> retry 3x into the same wall -> give up,
// possibly shipping a stale held note on a terminal turn). Note AdvisorRuntime.reset
// is still separately triggered by the PRIMARY's compaction / history rewrites;
// self-compaction is the advisor managing its OWN budget between those resets.
function textOf(content: Array<{ type: string; text?: string }>): string {
	return content.filter((c) => c.type === "text" && typeof c.text === "string").map((c) => c.text as string).join("");
}

// Render any tool-call argument value as readable text with REAL newlines preserved
// at EVERY depth. We never JSON.stringify content: that escapes every real newline
// into a literal backslash-n (so a heredoc body reaches the advisor as `<<'EOF'\n...`
// — the exact bug that produced a bogus "garbled markdown" advisory), and escaping
// only at the top level merely pushes the bug into nested strings (e.g. edits[].oldText).
// String leaves ride verbatim; containers are walked. Tool args are plain JSON data
// from the model, so there are no cycles or non-serializable leaves to guard against;
// a depth cap is the only (never-hit-in-practice) backstop.
function renderArgValue(v: unknown, indent: string, depth: number): string {
	// Multiline strings ride raw on following lines — NOT re-indented, which would
	// alter the very content (e.g. a heredoc body) the advisor must see verbatim.
	if (typeof v === "string") return v.includes("\n") ? `\n${v}` : ` ${v}`;
	if (v === null || typeof v !== "object") return ` ${String(v)}`;
	if (depth >= 8) return " […]";
	const childIndent = `${indent}  `;
	if (Array.isArray(v)) {
		if (v.length === 0) return " []";
		return v.map((e, i) => `\n${indent}- [${i}]${renderArgValue(e, childIndent, depth + 1)}`).join("");
	}
	const entries = Object.entries(v as Record<string, unknown>);
	if (entries.length === 0) return " {}";
	return entries.map(([k, val]) => `\n${indent}${k}:${renderArgValue(val, childIndent, depth + 1)}`).join("");
}

function renderToolArgs(args: Record<string, unknown> | undefined): string {
	if (!args || typeof args !== "object") return "";
	const entries = Object.entries(args);
	if (entries.length === 0) return "";
	return entries.map(([k, v]) => `${k}:${renderArgValue(v, "  ", 0)}`).join("\n");
}

// Format one primary turn (optionally preceded by the user prompt) as a markdown
// string with REAL newlines throughout (renderToolArgs keeps arg content verbatim).
// The sections are joined with explicit "\n\n" here so the boundary never depends on
// how a provider concatenates content parts — see buildReviewMessages.
export function formatTurnDelta(opts: {
	userPrompt?: string;
	assistant?: AssistantMessage;
	toolResults?: ToolResultMessage[];
}): string {
	const parts: string[] = [];
	if (opts.userPrompt?.trim()) parts.push(`#### User\n\n${opts.userPrompt.trim()}`);

	// Correlate calls → results by toolCallId so an edit's raw args can be suppressed
	// in favor of the result's diff — but ONLY when a SUCCESSFUL diff exists. A failed
	// edit (no diff, or an error result whose diff is untrustworthy) keeps its attempted
	// {oldText,newText} so the advisor can still diagnose the failure. Name-agnostic:
	// any non-error call whose result carries a diff.
	const diffByCallId = new Map<string, string>();
	for (const tr of opts.toolResults ?? []) {
		const id = (tr as { toolCallId?: string }).toolCallId;
		const d = (tr as { details?: { diff?: unknown } }).details?.diff;
		if (id && !tr.isError && typeof d === "string" && d.trim()) diffByCallId.set(id, d);
	}

	const a = opts.assistant;
	if (a) {
		const sub: string[] = [];
		for (const c of a.content) {
			if (c.type === "thinking" && c.thinking?.trim()) {
				sub.push(`<thinking>\n${c.thinking.trim()}\n</thinking>`);
			} else if (c.type === "text" && c.text?.trim()) {
				sub.push(c.text.trim());
			} else if (c.type === "toolCall") {
				// When this call produced a diff (a successful edit), suppress the raw
				// {oldText,newText} args and let the result's -/+ diff carry the change: the
				// args are two unannotated peer blobs and the advisor — reviewing AFTER the
				// edit landed (a fresh read shows the NEW side) — can't tell which is on disk
				// ("didn't persist"). With NO diff (failed edit, non-edit tool) show the args
				// verbatim; for a failed edit they're the only evidence of what was attempted.
				const edits = (c.arguments as { edits?: unknown[] } | undefined)?.edits;
				const hasDiff = diffByCallId.has((c as { id?: string }).id ?? "");
				if (hasDiff && Array.isArray(edits)) {
					const p = (c.arguments as { path?: string }).path ?? "?";
					sub.push(`→ tool \`${c.name}\`(${p}) — ${edits.length} block(s); diff in tool result`);
				} else {
					const argsText = renderToolArgs(c.arguments as Record<string, unknown> | undefined);
					sub.push(argsText ? `→ tool \`${c.name}\`:\n${argsText}` : `→ tool \`${c.name}\``);
				}
			}
		}
		if (sub.length) parts.push(`#### Assistant\n\n${sub.join("\n\n")}`);
	}

	for (const tr of opts.toolResults ?? []) {
		// Prefer the canonical line-numbered unified diff (the same view the human /
		// main model gets, computed by pi's edit-diff) for a SUCCESSFUL result: its -/+
		// markers unambiguously frame removed-vs-current lines, which the flat
		// {oldText,newText} echo lacks. It is also a pinned point-in-time snapshot of
		// THIS turn's change — the advisor's own read returns current (possibly later-
		// edited) disk, so the inline diff is not re-derivable and must ride verbatim.
		// On an ERROR, show the text body instead: the error is the diagnostic, and a
		// diff from a failed edit is untrustworthy (did it apply? partially?).
		const diff = (tr as { details?: { diff?: unknown } }).details?.diff;
		const body =
			!tr.isError && typeof diff === "string" && diff.trim()
				? diff
				: textOf(tr.content as Array<{ type: string; text?: string }>);
		parts.push(`#### Tool result: \`${tr.toolName}\`${tr.isError ? " (error)" : ""}\n\n${body || "(no text output)"}`);
	}
	return parts.join("\n\n");
}

// Assemble a review prompt as a BATCH of user messages: a header/reconfirm turn,
// then one user turn per primary-turn delta. Each message carries exactly ONE text
// block whose internal section separators ("\n\n") are explicit, so nothing depends
// on how a provider joins multiple content parts within a message. Between turns:
// OpenAI-family endpoints (OpenRouter, the default) keep them as distinct turns;
// Anthropic-family folds consecutive user turns into one (\n-joined, per the Messages
// API). Each turn starts with a #### / ### header, so it stays legible either way,
// and arg content rides verbatim (real newlines, no \n-escaping) — the whole point.
export function buildReviewMessages(preamble: string, batch: string[]): UserMessage[] {
	const now = Date.now();
	const messages: UserMessage[] = [
		{ role: "user", content: [{ type: "text", text: `### Session update\n\n${preamble}`.trimEnd() }], timestamp: now },
	];
	for (const delta of batch) {
		if (delta.trim()) messages.push({ role: "user", content: [{ type: "text", text: delta }], timestamp: now });
	}
	return messages;
}

// ---- build the persistent advisor Agent ----

function buildAdvisorAgent(opts: {
	cwd: string;
	model: Model<any>;
	thinkingLevel: string;
	systemPrompt: string;
	modelRegistry: any;
	adviseTool: AdviseTool;
}): Agent {
	const readOnly = createReadOnlyTools(opts.cwd);
	const thinkingLevel = opts.model.reasoning ? (opts.thinkingLevel as any) : ("off" as any);
	return new Agent({
		initialState: {
			systemPrompt: opts.systemPrompt,
			model: opts.model,
			thinkingLevel,
			tools: [opts.adviseTool, ...readOnly] as any,
		},
		convertToLlm,
		// Use the bundled default streamFn (pi-agent-core's own streamSimple); we
		// only supply auth. The `@earendil-works/pi-ai` extension surface does not
		// expose streamSimple, so a custom streamFn is not an option here.
		getApiKey: (provider: string) => opts.modelRegistry.getApiKeyForProvider(provider),
	});
}

// ---- AdvisorRuntime — drives the advisor agent off primary turn deltas ----

/**
 * Feeds the persistent advisor agent one delta per primary turn, serialized so
 * the agent is never prompted while already streaming. On context overflow (or
 * any history rewrite) the caller invokes `reset()`, which clears the advisor's
 * own context so the next delta replays fresh.
 */
export class AdvisorRuntime {
	#pending: string[] = [];
	#held: AdvisorNote[] = [];
	// Keys re-raised during the in-flight review; drives the post-review prune.
	#reraised: Set<string> | undefined;
	// Outcome of the most recently completed drain batch: "ok" (successful review)
	// or "failed" (errored 3x and dropped). Lets waitUntilSettled distinguish a
	// genuine settle from a give-up, so held notes aren't delivered as if confirmed.
	#lastOutcome: "ok" | "failed" | undefined;
	// Epoch of the in-flight review; advice callbacks are honored only while it still
	// matches #epoch. A reset/dispose bumps #epoch, orphaning a stale review whose
	// late advise() calls would otherwise leak into the moved-on session.
	#reviewEpoch = -1;
	#settleWaiters: Array<{ settle: () => void; cancel: () => void }> = [];
	#busy = false;
	#backlog = 0;
	#failures = 0;
	#epoch = 0;
	// Lifetime input/output/cost from advisor turns already discarded by a
	// self-compaction (#softReset). The agent's message list only holds the CURRENT
	// (post-compaction) context, so without folding these in, /advisor status would
	// undercount lifetime tokens/cost after each self-compaction. A full reset()
	// (primary compaction / new session) zeroes them — that is a fresh accounting.
	#cumInput = 0;
	#cumOutput = 0;
	#cumCost = 0;
	disposed = false;

	// Self-compact when the advisor's own context reaches this % of its window
	// (proactively, before the next review prompt). Below 100 so a fresh replay of
	// the next batch comfortably fits; the reactive stopReason=="length" path is the
	// backstop if a single batch crosses it anyway.
	private readonly compactAtPercent: number;

	constructor(
		private readonly agent: Agent,
		private readonly adviseTool: AdviseTool,
		private readonly retryDelayMs = 1000,
		private readonly onDebug?: (...a: unknown[]) => void,
		compactAtPercent = 80,
	) {
		this.compactAtPercent = compactAtPercent;
	}

	/**
	 * Self-compaction: clear ONLY the advisor agent's own message history,
	 * preserving the pending queue, held notes, backlog, failure count, and settle
	 * waiters. Safe because the agent transcript is a pure linear accumulation of
	 * independent turn deltas — no essential cross-turn state lives there (held
	 * notes ride the reconfirm preamble). Unlike reset(), this does NOT bump the
	 * epoch (the in-flight review is ours, not orphaned) nor drop queued/held work.
	 */
	#softReset(): void {
		// Preserve lifetime token/cost accounting before the about-to-be-cleared
		// messages are gone (see #cumInput/#cumOutput/#cumCost).
		for (const m of this.agent.state.messages) {
			if (m.role === "assistant" && (m as AssistantMessage).usage) {
				const u = (m as AssistantMessage).usage;
				this.#cumInput += u.input ?? 0;
				this.#cumOutput += u.output ?? 0;
				this.#cumCost += u.cost?.total ?? 0;
			}
		}
		try {
			this.agent.abort();
		} catch {}
		try {
			this.agent.reset();
		} catch {}
	}

	get backlog(): number {
		return this.#backlog;
	}

	/** True when no batch is in flight and nothing is queued: the advisor has
	 *  reviewed everything pushed so far ("settled"). */
	get idle(): boolean {
		return !this.#busy && this.#pending.length === 0;
	}

	/** Whether any high-severity note is currently held awaiting reconfirmation. */
	get hasHeld(): boolean {
		return this.#held.length > 0;
	}

	/**
	 * Stash a high-severity note for reconfirmation. It rides the next review as a
	 * reconfirm preamble (see `#drain`); survivors are taken via `takeHeld()` and
	 * steered in by the catch-up block once the advisor settles. Deduped by note
	 * text so re-raising during a reconfirm doesn't pile up duplicates; the re-raise
	 * is recorded so the post-review prune keeps it.
	 */
	hold(note: string, severity?: AdvisorSeverity): void {
		if (this.disposed) return;
		const key = dedupeKey(note);
		this.#reraised?.add(key);
		const existing = this.#held.find((n) => dedupeKey(n.note) === key);
		if (!existing) {
			this.#held.push({ note, severity });
		} else if (rankOf(severity) > rankOf(existing.severity)) {
			// Honor an escalation (e.g. a held concern re-raised as a blocker).
			existing.severity = severity;
		}
	}

	/** Remove and return the currently-held notes (the reconfirmation survivors). */
	takeHeld(): AdvisorNote[] {
		return this.#held.splice(0);
	}

	/** Whether advice from the in-flight review is still valid (not orphaned by a
	 *  reset/dispose). The delivery layer consults this to drop late stale callbacks. */
	get acceptingAdvice(): boolean {
		return !this.disposed && this.#reviewEpoch === this.#epoch;
	}

	/**
	 * If `note` matches a currently-held note, count it as a reconfirmation (so the
	 * post-review prune keeps it) and return true. Lets the delivery layer suppress a
	 * de-escalation (a held blocker re-raised as a nit) instead of dropping the
	 * blocker and shipping a nit.
	 */
	reconfirmIfHeld(note: string): boolean {
		const key = dedupeKey(note);
		if (!this.#held.some((n) => dedupeKey(n.note) === key)) return false;
		this.#reraised?.add(key);
		return true;
	}

	/**
	 * Resolve once the advisor has caught up (`idle`), or `timeoutMs` elapses, or
	 * `signal` aborts. Drives the per-turn catch-up block. Resolves "settled"
	 * immediately if already idle/disposed.
	 */
	waitUntilSettled(timeoutMs: number, signal?: AbortSignal): Promise<"settled" | "timeout" | "aborted" | "failed"> {
		if (this.disposed) return Promise.resolve("aborted");
		if (this.idle) return Promise.resolve(this.#lastOutcome === "failed" ? "failed" : "settled");
		return new Promise((resolve) => {
			let done = false;
			let waiter: { settle: () => void; cancel: () => void };
			let timer: ReturnType<typeof setTimeout>;
			const finish = (r: "settled" | "timeout" | "aborted" | "failed") => {
				if (done) return;
				done = true;
				clearTimeout(timer);
				const i = this.#settleWaiters.indexOf(waiter);
				if (i >= 0) this.#settleWaiters.splice(i, 1);
				signal?.removeEventListener("abort", onAbort);
				resolve(r);
			};
			const onAbort = () => finish("aborted");
			waiter = {
				// Fired when the drain reaches idle (a review completed).
				settle: () => {
					if (this.disposed) finish("aborted");
					else if (this.idle) finish(this.#lastOutcome === "failed" ? "failed" : "settled");
				},
				// Fired by reset()/dispose(): resolve immediately rather than waiting for
				// the in-flight prompt to unwind (which could take up to the timeout).
				cancel: () => finish("aborted"),
			};
			timer = setTimeout(() => finish("timeout"), timeoutMs);
			this.#settleWaiters.push(waiter);
			if (signal) {
				if (signal.aborted) finish("aborted");
				else signal.addEventListener("abort", onAbort);
			}
		});
	}

	#notifySettled(): void {
		for (const w of [...this.#settleWaiters]) w.settle();
	}

	/** Resolve all pending waiters as "aborted" (used by reset/dispose). */
	#cancelWaiters(): void {
		for (const w of [...this.#settleWaiters]) w.cancel();
	}

	get usage(): { input: number; output: number; cost: number; contextTokens: number; contextPercent: number | null } {
		let input = this.#cumInput;
		let output = this.#cumOutput;
		let cost = this.#cumCost;
		let contextTokens = 0;
		for (const m of this.agent.state.messages) {
			if (m.role === "assistant" && (m as AssistantMessage).usage) {
				const u = (m as AssistantMessage).usage;
				input += u.input ?? 0;
				output += u.output ?? 0;
				cost += u.cost?.total ?? 0;
				// Latest request's input + cache reads ≈ current advisor context size.
				contextTokens = (u.input ?? 0) + (u.cacheRead ?? 0) + (u.cacheWrite ?? 0);
			}
		}
		const window = (this.agent.state.model as { contextWindow?: number } | undefined)?.contextWindow;
		const contextPercent = window ? Math.round((contextTokens / window) * 100) : null;
		return { input, output, cost, contextTokens, contextPercent };
	}

	/** Queue a rendered primary-turn delta (markdown string) for review. */
	push(deltaText: string): void {
		if (this.disposed || !deltaText.trim()) return;
		this.#pending.push(deltaText);
		this.#backlog++;
		void this.#drain();
	}

	/** Re-prime after a history rewrite (compaction / session switch / fork). */
	reset(): void {
		this.#epoch++;
		this.#pending = [];
		this.#held = [];
		this.#reraised = undefined;
		this.#lastOutcome = undefined;
		this.#backlog = 0;
		this.#failures = 0;
		// Full reset = fresh accounting (unlike #softReset, which preserves these).
		this.#cumInput = this.#cumOutput = this.#cumCost = 0;
		this.adviseTool.resetDelivered();
		try {
			this.agent.abort();
		} catch {}
		try {
			this.agent.reset();
		} catch {}
		this.#cancelWaiters();
	}

	dispose(): void {
		this.disposed = true;
		this.#epoch++;
		this.#pending = [];
		this.#held = [];
		this.#reraised = undefined;
		this.#lastOutcome = undefined;
		this.#backlog = 0;
		try {
			this.agent.abort();
		} catch {}
		this.#cancelWaiters();
	}

	async #drain(): Promise<void> {
		if (this.#busy) return;
		this.#busy = true;
		try {
			while (!this.disposed && this.#pending.length) {
				const batch = this.#pending.splice(0);
				const turns = batch.length;
				// Rough gauge of how many turns are still unreviewed (status display only).
				this.#backlog = Math.max(0, this.#backlog - turns);
				const epoch = this.#epoch;
				// Re-offer held notes as a reconfirm preamble WITHOUT removing them from
				// #held: hasHeld/takeHeld must stay accurate while this review is in flight
				// (the catch-up block reads them concurrently — `push()` runs `#drain` up to
				// the first await, so a splice here would empty #held before the block even
				// looks). After a successful review we prune any offered note the advisor
				// did NOT re-raise (it's been resolved).
				const offered = [...this.#held];
				const offeredKeys = new Set(offered.map((n) => dedupeKey(n.note)));
				const preamble = formatReconfirmPreamble(offered);
				this.#reraised = new Set();
				this.#reviewEpoch = epoch;
				const messages = buildReviewMessages(preamble, batch);
				const promptChars = messages.reduce(
					(n, m) =>
						n +
						(Array.isArray(m.content)
							? m.content.reduce(
									(k: number, b: { type: string; text?: string }) => k + (b.type === "text" ? (b.text?.length ?? 0) : 0),
									0,
								)
							: 0),
					0,
				);
				// A review "fails" either by throwing OR — the common case — by resolving
				// with an assistant message whose stopReason is "error"/"aborted" (the agent
				// loop records provider failures that way instead of throwing). A failed
				// review must NOT prune held notes (we'd drop them as if recanted).
				let failed = false;
				// PROACTIVE self-compaction: if our own context has crossed the budget,
				// clear the agent history now so this batch replays into a fresh context
				// (held notes survive via the reconfirm preamble) instead of marching into
				// an overflow. Skipped when already fresh (nothing to reclaim).
				const pct = this.usage.contextPercent;
				if (pct !== null && pct >= this.compactAtPercent && this.agent.state.messages.length > 0) {
					this.onDebug?.("advisor self-compacting (proactive), ctx=", pct, "% >=", this.compactAtPercent, "%");
					this.#softReset();
				}
				let stale = false;
				try {
					// Inner loop: at most ONE reactive self-compaction retry. If the
					// advisor's own context overflows mid-review (stopReason "length"), clear
					// its history and replay THIS batch into a fresh context instead of
					// counting a failure and retrying 3x into the same wall. Loop-safe: a
					// fresh replay that STILL overflows means the single batch genuinely
					// doesn't fit, so it falls through to the failed handling below.
					let last: AssistantMessage | undefined;
					for (let attempt = 0; attempt < 2; attempt++) {
						this.onDebug?.("prompting advisor agent, delta chars=", promptChars, "held=", offered.length);
						await this.agent.prompt(messages);
						if (this.#epoch !== epoch) {
							stale = true;
							break; // reset/dispose during the prompt; batch is stale
						}
						last = this.agent.state.messages[this.agent.state.messages.length - 1] as AssistantMessage;
						if (last?.stopReason === "length" && attempt === 0) {
							this.onDebug?.("advisor context overflow, self-compacting (reactive) and replaying batch fresh");
							this.#softReset();
							// Roll back any concern/blocker the DISCARDED overflowed attempt held:
							// it was raised against a truncated view, and offeredKeys was snapshotted
							// pre-attempt so the success-prune below can't reach it — left in place
							// it would later deliver (e.g. terminal best-effort) as if confirmed.
							// The fresh replay re-raises it if genuine; otherwise it's correctly
							// gone. Exact: #held only GROWS during an attempt (hold() never removes),
							// so restoring the pre-batch snapshot drops exactly the attempt's adds.
							// (Nits were already steered + recorded in #delivered, which survives
							// softReset, so the replay dedupes them — no double-fire.)
							this.#held = offered.slice();
							this.#reraised = new Set();
							continue;
						}
						break;
					}
					if (stale) {
						this.#reraised = undefined;
						continue;
					}
					if (last?.stopReason === "error" || last?.stopReason === "aborted" || last?.stopReason === "length") {
						// error/aborted = provider failure (recorded, not thrown); length =
						// truncated review (a fresh replay still didn't fit) — in all three the
						// advisor didn't finish, so don't prune held notes on its accidental
						// "silence".
						this.onDebug?.("advisor review incomplete, stop=", last?.stopReason, "err=", last?.errorMessage ?? "-");
						failed = true;
					} else {
						// Success: prune recanted holds (offered notes the advisor stayed silent on).
						for (const key of offeredKeys) {
							if (!this.#reraised?.has(key)) {
								const i = this.#held.findIndex((n) => dedupeKey(n.note) === key);
								if (i >= 0) this.#held.splice(i, 1);
							}
						}
						this.#lastOutcome = "ok";
						this.#failures = 0;
						this.onDebug?.("advisor turn done, stop=", last?.stopReason);
					}
					this.#reraised = undefined;
				} catch (e) {
					this.#reraised = undefined;
					this.onDebug?.("advisor prompt threw", String(e));
					// A reset/dispose aborts the in-flight prompt; drop the stale batch.
					// Held notes were never removed, so nothing to restore there.
					if (this.#epoch !== epoch) continue;
					failed = true;
				}
				if (failed) {
					this.#failures++;
					if (this.#failures >= 3) {
						// Gave up reconfirming this batch. Mark failed so waitUntilSettled
						// reports it (don't deliver held notes as if confirmed).
						this.#failures = 0;
						this.#lastOutcome = "failed";
					} else {
						this.#pending.unshift(...batch);
						this.#backlog += turns;
						await new Promise((r) => setTimeout(r, this.retryDelayMs));
					}
				}
			}
		} finally {
			this.#busy = false;
			if (this.idle) this.#notifySettled();
		}
	}
}

// ===========================================================================
// Extension wiring
// ===========================================================================

const ADVISORY_TYPE = "advisory";
// Footer status key. Statuses are ordered alphabetically by key; "q-advisor"
// sorts after "permissions"/"provider-system-prompt" but before "sub-bar", so
// Advisor shows as a middle segment. Change this to reposition it (e.g.
// "a-advisor" for leftmost).
const STATUS_KEY = "q-advisor";
const DEBUG = !!process.env.ADVISOR_DEBUG;
const dbg = (...a: unknown[]) => {
	if (DEBUG) console.error("[advisor]", ...a);
};
const BLOCK_BASE_MS = 15_000;
const BLOCK_CAP_MS = 120_000;

// Set by the handoff extension (pi-amplike) via the same Symbol.for key while a
// handoff is in flight — from the moment it becomes pending until the new
// session's prompt has been dispatched. During that window the primary session
// is being torn down / replaced and its deferred handoff prompt is racing to be
// sent, so the advisor must not inject messages or (worse) trigger an
// autonomous turn: doing so either crashes the handoff ("Agent is already
// processing") or leaks a stray advisory into the brand-new session.
const HANDOFF_IN_PROGRESS_KEY = Symbol.for("pi-amplike-handoff-in-progress");
function handoffInProgress(): boolean {
	return !!(globalThis as any)[HANDOFF_IN_PROGRESS_KEY];
}

// Emitted by the handoff extension after its tool path replaces the session
// transcript via the low-level sessionManager.newSession() (which emits no
// session_start). Must match HANDOFF_SESSION_REPLACED_CHANNEL in handoff.ts.
const HANDOFF_SESSION_REPLACED_CHANNEL = "pi-amplike:handoff-session-replaced";
const DEFAULT_ADVISOR_PROVIDER = "openrouter";
const DEFAULT_ADVISOR_MODEL = "z-ai/glm-5.2";
const DEFAULT_THINKING = "low";

function agentDir(): string {
	const env = process.env.PI_CODING_AGENT_DIR;
	if (env) return env.startsWith("~/") ? path.join(os.homedir(), env.slice(2)) : env;
	return path.join(os.homedir(), ".pi", "agent");
}

const STATE_FILE = () => path.join(agentDir(), ".advisor-state.json");

function loadEnabled(): boolean {
	// Opt-out: enabled unless explicitly turned off (`/advisor off`).
	try {
		return JSON.parse(fs.readFileSync(STATE_FILE(), "utf8")).enabled !== false;
	} catch {
		return true;
	}
}
function saveEnabled(enabled: boolean): void {
	try {
		fs.writeFileSync(STATE_FILE(), JSON.stringify({ enabled }), "utf8");
	} catch {}
}

// Default advisor system prompt, bundled so the package is self-contained. A
// user-provided ~/.pi/agent/system-prompts/advisor.md overrides it when present.
const DEFAULT_ADVISOR_SYSTEM_PROMPT = `You bring a different angle, and advocate for the user and for code-quality & robustness.
You're watching over a main coding agent as a peer programmer:
- They might not have thought about an edge case, or realized a more elegant approach exists.
- They might be sinking deeper into a hole that will not accomplish the user's request.

Your job is to offer that view before they sink work into the wrong direction.

<scope>
You critique the agent's work; you never do it yourself. You are not a participant
in the conversation and never address the user. When the agent answers a question
or explains something, your job is to check THAT answer for errors — not to research
or compose your own answer. If the agent is sound, stay SILENT. Never try to fulfill
the user's request yourself; that is the agent's job, not yours.
</scope>

<workflow>
You receive the agent's transcript incrementally, including their thoughts and tool calls/results.
You have read-only access through \`read\`, \`grep\`, \`find\` to verify your suspicions.
Keep exploration lean:
- 2–3 tool calls per advise, at most.
- Exception: a critical bug may need deeper verification before raising a blocker.
</workflow>

<communication>
- You call \`advise\` to surface commentary to the driving agent; at most one \`advise\` per update
  (exception: when reconfirming held advisories, re-raise EACH one that still applies).
- Prefer SILENCE when the agent is on track. Most updates should produce no advice at all.
- \`advise\` is for ACTIONABLE advice ONLY. NEVER use it to report status, acknowledge,
  confirm, summarize, or signal "all clear" / "resolved" / "nothing further needed" /
  "looks good". If you have nothing for the agent to DO, emit nothing — silence is the
  signal that all is well. A held advisory that no longer applies is dropped by staying
  silent, NOT by announcing it's resolved.
- Address the agent directly. Offer alternatives, not lectures.
- NEVER restate information the agent already has, including errors they already saw
  (type errors, LSP diagnostics, failed builds, failing tests, lint output).
- NEVER repeat advice you already gave, and NEVER send the same advice twice. (Re-raising a
  held advisory you are explicitly asked to reconfirm is NOT a repeat.)
- NEVER nitpick about things the user already stated they are okay with. You advocate for the user.
</communication>

<critical>
A low-confidence bar applies ONLY to concrete technical risk.
Generic uncertainty, vague unease, or user-intent ambiguity → stay SILENT.

NEVER second-guess decisions the agent understands and is committed to, unless you are certain.

NEVER advise on intent or process:
- Do not push the agent to ask for clarification, confirm scope, or summarize before acting.
- Do not question whether the user's ask is clear enough.
- Intent is the agent's domain; it defaults to informed action.
- Your lane: correctness, edge cases, design, robustness.

Cite the exact instruction or risk.
</critical>

<severity>
**nit** (or omitted)
- Non-urgent cleanup, refactor, style, simplification, a missed-but-minor opportunity.
- Low-stakes: surfaced to the agent without stalling or throttling its work.

**concern**
- The agent might be heading the wrong way or missed something material.
- Exploring the wrong code path, picking a fragile approach when a better one exists,
  missing a constraint, or about to bake in a bad edge case.
- Offers your view; the agent decides.

**blocker**
- Stop and reconsider. Use ONLY when continuing will clearly:
  - Waste the user's time with a larger wrong refactor, or
  - Force the user to interrupt later because the agent is going in circles, or
  - Produce something fundamentally unsound.
- Verify thoroughly before raising.

concern/blocker are held and reconfirmed before they reach the agent: you may be
shown your held advisories again alongside newer activity. Re-raise EACH that still
applies (same severity, or higher if it's gotten worse — never lower) — this is not a
repeat, and re-raising several is fine here. Stay silent on any the agent has since
addressed; silence drops them.
</severity>

You MAY suggest an approach or fix if you've explored enough to be confident.
Offer the better design, not just the warning.
`;

function loadSystemPrompt(cwd: string): string {
	let prompt = "";
	try {
		prompt = fs.readFileSync(path.join(agentDir(), "system-prompts", "advisor.md"), "utf8");
	} catch {
		prompt = DEFAULT_ADVISOR_SYSTEM_PROMPT;
	}
	// Append WATCHDOG.md (advisor-only project guidance) if present in cwd.
	try {
		const wd = fs.readFileSync(path.join(cwd, "WATCHDOG.md"), "utf8").trim();
		if (wd) prompt += `\n\nEspecially pay attention to:\n<attention>\n${wd}\n</attention>`;
	} catch {}
	return prompt;
}

export default function (pi: ExtensionAPI) {
	let enabled = loadEnabled();

	// Lazily-built advisor state, rebuilt when cwd/model changes or session resets.
	let runtime: AdvisorRuntime | undefined;
	let activeModelLabel: string | undefined;
	let builtForCwd: string | undefined;

	// Delta accumulation across the lifecycle.
	let pendingUserPrompt: string | undefined;

	// The advise tool bound to the live runtime (held so the catch-up block can
	// mark held notes delivered at the actual delivery point).
	let adviseTool: AdviseTool | undefined;

	// Consecutive mid-run catch-up blocks, for the backoff (reset when the advisor
	// settles or a turn doesn't need to block).
	let consecutiveBlocks = 0;

	// Set when the user aborts (Escape) around a catch-up block: while true, late
	// advisor advice is delivered WITHOUT triggerTurn so it can't auto-resume the run
	// the user just stopped. Cleared when the user drives the next turn.
	let autoResumeSuppressed = false;

	// Whether the primary is currently stopped, having returned a final answer (the
	// most recent turn was terminal and the user hasn't driven a new one yet). When
	// true, advice we steer in wakes the stopped agent for a follow-up turn, so we
	// tell it to reply with a fresh, self-contained final answer rather than a
	// back-and-forth the user must stitch together. Set at turn_end (before any
	// await, since late nit callbacks read it) and cleared at before_agent_start.
	let currentTurnTerminal = false;

	// ---- statusbar: minimalistic per-session advisor cost ----
	// Reflects the live advisor lifetime cost (rt.usage.cost) in the footer status
	// bar as `│ Advisor: $N`. Cleared when the advisor is off or torn down.
	//
	// Footer ordering: pi sorts extension statuses alphabetically BY KEY and joins
	// them with a single space (no separators of its own). So the key controls
	// position and we draw our own `│` divider in the text. STATUS_KEY sorts after
	// "permissions"/"provider-system-prompt" but before "sub-bar", placing Advisor as
	// a middle segment rather than the leftmost.
	//
	// LEADING bar only (no trailing): whatever follows draws its own separator
	// (e.g. pi-sub-bar with statusLeadingDivider:true starts with `│`), so a trailing
	// bar here would double up (`│ Advisor │ │ …`).
	function updateStatus(ctx: unknown): void {
		const ui = (ctx as {
			ui?: { setStatus?: (k: string, t: string | undefined) => void; theme?: { fg: (c: string, s: string) => string } };
		}).ui;
		if (!ui?.setStatus) return;
		if (!enabled || !runtime) {
			ui.setStatus(STATUS_KEY, undefined);
			return;
		}
		const bar = ui.theme ? ui.theme.fg("dim", "│") : "│";
		ui.setStatus(STATUS_KEY, `${bar} Advisor: $${runtime.usage.cost.toFixed(2)}`);
	}

	// ---- advice delivery into the primary session ----
	// Called synchronously by the advise tool during a review. Returns true if the
	// note was delivered now (recorded for dedup), false if held for reconfirmation
	// (left unrecorded so it can re-fire; the catch-up block delivers survivors).
	function deliverAdvice(note: string, severity?: AdvisorSeverity): boolean {
		// Stand down entirely while a handoff is being performed (see comment on
		// HANDOFF_IN_PROGRESS_KEY). Report it as "delivered" so it isn't held into
		// the brand-new session.
		if (handoffInProgress()) {
			dbg("handoff in progress, dropping advice", severity, JSON.stringify(note).slice(0, 80));
			return true;
		}
		// Drop late callbacks the session has moved past: advisor turned off, or a
		// reset/dispose orphaned the in-flight review (its epoch no longer matches).
		// Report "delivered" so AdviseTool doesn't keep re-firing it.
		if (!enabled || (runtime && !runtime.acceptingAdvice)) {
			dbg("dropping stale/disabled advice", severity, JSON.stringify(note).slice(0, 80));
			return true;
		}

		if (isHighSeverity(severity)) {
			// Always held: the advisor is seconds behind, so any high-severity advice
			// is about a state the agent has likely moved past. Reconfirm + the catch-up
			// block deliver it (against unraced state) only if it still applies.
			dbg("deliverAdvice hold", severity, JSON.stringify(note).slice(0, 120));
			runtime?.hold(note, severity);
			return false;
		}

		// A nit whose text matches a held high-severity note is a de-escalation the
		// prompt forbids; treat it as a reconfirmation (keep the held note at its
		// severity) instead of shipping a nit and pruning the blocker.
		if (runtime?.reconfirmIfHeld(note)) {
			dbg("nit reconfirms a held note; keeping it held", JSON.stringify(note).slice(0, 120));
			return true;
		}

		// nit: deliver now, tagged as raised about an earlier step. triggerTurn wakes
		// an idle agent — unless the user just aborted (Escape), in which case we must
		// not auto-resume the run they stopped.
		dbg("deliverAdvice nit", JSON.stringify(note).slice(0, 120));
		const notes: AdvisorNote[] = [{ note, severity }];
		const content = formatAdvisoryContent(notes, { stale: true, finalAnswer: currentTurnTerminal });
		pi.sendMessage({ customType: ADVISORY_TYPE, content, display: true, details: { notes } }, { deliverAs: "steer", triggerTurn: !autoResumeSuppressed });
		return true;
	}

	// ---- steer held survivors into the primary (called by the catch-up block) ----
	function deliverHeld(notes: AdvisorNote[], opts?: { terminal?: boolean }): void {
		if (handoffInProgress() || !notes.length) return;
		for (const n of notes) {
			dbg("deliverHeld", n.severity, JSON.stringify(n.note).slice(0, 120));
			const content = formatAdvisoryContent([n], { finalAnswer: !!opts?.terminal });
			pi.sendMessage({ customType: ADVISORY_TYPE, content, display: true, details: { notes: [n] } }, { deliverAs: "steer", triggerTurn: !autoResumeSuppressed });
			// Record at the real delivery point (onAdvice→false never recorded it), so a
			// later same-or-lower-severity repeat is deduped.
			adviseTool?.markDelivered(n.note, n.severity);
		}
	}

	function teardown(): void {
		runtime?.dispose();
		runtime = undefined;
		adviseTool = undefined;
		activeModelLabel = undefined;
		builtForCwd = undefined;
		pendingUserPrompt = undefined;
		consecutiveBlocks = 0;
		autoResumeSuppressed = false;
		currentTurnTerminal = false;
	}

	// Re-prime for a replaced transcript without tearing down the advisor agent:
	// clear its context so the next delta replays fresh. Used by both the
	// session_start handler and the handoff session-replaced signal so the two
	// paths can't drift.
	function resetAdvisorState(): void {
		runtime?.reset();
		pendingUserPrompt = undefined;
		consecutiveBlocks = 0;
		autoResumeSuppressed = false;
		currentTurnTerminal = false;
	}

	// ---- build the advisor agent lazily (needs ctx for model/registry/cwd) ----
	async function ensureRuntime(ctx: {
		cwd: string;
		modelRegistry: any;
		model: any;
	}): Promise<AdvisorRuntime | undefined> {
		if (runtime && builtForCwd === ctx.cwd) return runtime;
		if (runtime && builtForCwd !== ctx.cwd) teardown();

		if (!ctx.model) return undefined;

		// Resolve advisor model: modes.json "advisor" first, else the default.
		let model: any;
		let thinkingLevel = DEFAULT_THINKING;
		try {
			const resolved = await resolveModelAndThinking(ctx.cwd, ctx.modelRegistry, ctx.model, DEFAULT_THINKING, {
				mode: "advisor",
			});
			// resolveModelAndThinking falls back to the current model when "advisor"
			// mode is absent; detect that and use our hard default instead.
			const sameAsPrimary = resolved.model === ctx.model;
			model = sameAsPrimary ? undefined : resolved.model;
			thinkingLevel = resolved.thinkingLevel || DEFAULT_THINKING;
		} catch {}
		if (!model) {
			model = ctx.modelRegistry.find(DEFAULT_ADVISOR_PROVIDER, DEFAULT_ADVISOR_MODEL);
		}
		if (!model) return undefined;

		adviseTool = new AdviseTool(deliverAdvice);
		const agent = buildAdvisorAgent({
			cwd: ctx.cwd,
			model,
			thinkingLevel,
			systemPrompt: loadSystemPrompt(ctx.cwd),
			modelRegistry: ctx.modelRegistry,
			adviseTool,
		});
		// ADVISOR_COMPACT_AT: % of the advisor's context window at which it self-
		// compacts (clamped 50..95; default 80).
		const compactAt = Math.min(95, Math.max(50, Number(process.env.ADVISOR_COMPACT_AT) || 80));
		runtime = new AdvisorRuntime(agent, adviseTool, 1000, dbg, compactAt);
		activeModelLabel = `${model.provider}/${model.id}`;
		builtForCwd = ctx.cwd;
		dbg("built advisor runtime, model=", activeModelLabel);
		return runtime;
	}

	// ---- event wiring ----

	// Capture the user prompt so it rides the next turn delta to the advisor.
	pi.on("before_agent_start", (event) => {
		if (!enabled) return;
		// The user is driving a new turn; clear any post-abort auto-resume suppression.
		autoResumeSuppressed = false;
		// The agent is now working, not stopped — drop any leftover terminal flag so a
		// late nit from an earlier still-draining review can't wrongly claim the user
		// already has a final answer. (Re-set accurately at this turn's turn_end.)
		currentTurnTerminal = false;
		pendingUserPrompt = event.prompt;
	});

	// One delta per primary turn (assistant message + its tool results). After
	// pushing, run the catch-up block: this hook is awaited by the agent loop, so
	// awaiting here stalls the primary's next step until the advisor catches up.
	pi.on("turn_end", async (event, ctx) => {
		// Test seam: skip live model review (keeps the /advisor test delivery path) so
		// nit delivery can be tested in the pi harness without the advisor model.
		if (!enabled || process.env.ADVISOR_NO_REVIEW) return;

		// Record terminality BEFORE any await: a nit from an EARLIER, still-draining
		// review can fire its deliverAdvice callback during the ensureRuntime() await
		// below, and it reads this flag — so it must already reflect the just-ended
		// turn. Semantics: "the primary is now stopped, having returned a final answer"
		// (set true on a terminal turn, false otherwise; cleared when the user drives a
		// new turn). Used so advice that wakes the stopped agent appends the
		// self-contained-final-answer guidance.
		const terminal = isTerminalTurn(event.message as any);
		currentTurnTerminal = terminal;

		const rt = await ensureRuntime(ctx as any);
		dbg("turn_end", "enabled=", enabled, "runtime=", !!rt, "model=", activeModelLabel);
		if (!rt) return;

		const delta = formatTurnDelta({
			userPrompt: pendingUserPrompt,
			assistant: event.message as AssistantMessage,
			toolResults: event.toolResults as ToolResultMessage[],
		});
		pendingUserPrompt = undefined;
		rt.push(delta);

		// Don't block during a handoff teardown (we'd stall the replacement).
		if (handoffInProgress()) return;
		updateStatus(ctx);
		consecutiveBlocks = await runTurnBlock({
			terminal,
			runtime: rt,
			consecutiveBlocks,
			baseMs: BLOCK_BASE_MS,
			capMs: BLOCK_CAP_MS,
			signal: (ctx as any).signal,
			notify: (m) => {
				try {
					(ctx as any).ui?.notify?.(m, "info");
				} catch {}
			},
			deliverHeld,
		});
		// If the user aborted (Escape) around the block, suppress auto-resume so a late
		// advisor callback from the still-running review can't restart the stopped run.
		if ((ctx as any).signal?.aborted) autoResumeSuppressed = true;
		// Refresh the footer cost after the advisor caught up (review cost is now in).
		updateStatus(ctx);
	});

	// Re-prime the advisor when the primary transcript is rewritten.
	pi.on("session_compact", (_event, ctx) => {
		runtime?.reset();
		updateStatus(ctx);
	});
	pi.on("session_start", (event, ctx) => {
		// new/resume/fork replace history; a plain startup/reload keeps it.
		if (event.reason === "new" || event.reason === "resume" || event.reason === "fork") {
			resetAdvisorState();
			updateStatus(ctx);
		}
	});

	// Tool-path handoff replaces the transcript without a session_start event
	// (low-level sessionManager.newSession()), so reset off this explicit signal.
	pi.events.on(HANDOFF_SESSION_REPLACED_CHANNEL, () => resetAdvisorState());

	pi.on("session_shutdown", (_event, ctx) => {
		teardown();
		(ctx as { ui?: { setStatus?: (k: string, t: string | undefined) => void } }).ui?.setStatus?.(STATUS_KEY, undefined);
	});

	// ---- advisory card rendering ----
	pi.registerMessageRenderer<{ notes: AdvisorNote[] }>(ADVISORY_TYPE, (message, _options, theme) => {
		const notes = message.details?.notes;
		if (!notes?.length) return undefined;
		const container = new Container();
		for (const n of notes) {
			const color = n.severity === "blocker" ? "error" : n.severity === "concern" ? "warning" : "dim";
			const tag = (n.severity ?? "nit").toUpperCase();
			container.addChild(new Text(`${theme.fg(color, `◆ advisor [${tag}]`)} ${theme.fg("muted", n.note)}`, 1, 0));
		}
		return container;
	});

	// ---- /advisor command ----
	pi.registerCommand("advisor", {
		description: "Toggle/inspect the advisor (a second model that reviews each turn). Usage: /advisor [on|off|status]",
		handler: async (args, ctx) => {
			const arg = args.trim().toLowerCase();

			if (arg === "status" || arg === "") {
				const state = enabled ? "enabled" : "disabled";
				if (!enabled) {
					ctx.ui.notify(`advisor ${state}`, "info");
					updateStatus(ctx);
					return;
				}
				const rt = await ensureRuntime(ctx as any);
				if (!rt) {
					ctx.ui.notify(`advisor enabled but no advisor model is available`, "warning");
					return;
				}
				updateStatus(ctx);
				const u = rt.usage;
				const ctxStr = u.contextPercent !== null ? `${u.contextPercent}% (${u.contextTokens} tok)` : `${u.contextTokens} tok`;
				ctx.ui.notify(
					`advisor ${state} — model ${activeModelLabel}, backlog ${rt.backlog}, ` +
						`tokens ${u.input}in/${u.output}out, cost $${u.cost.toFixed(4)}, ctx ${ctxStr}`,
					"info",
				);
				return;
			}

			if (arg === "on") {
				enabled = true;
				saveEnabled(true);
				const rt = await ensureRuntime(ctx as any);
				updateStatus(ctx);
				ctx.ui.notify(rt ? `advisor on — ${activeModelLabel}` : `advisor on, but no advisor model available`, rt ? "info" : "warning");
				return;
			}
			if (arg === "off") {
				enabled = false;
				saveEnabled(false);
				teardown();
				updateStatus(ctx);
				ctx.ui.notify("advisor off", "info");
				return;
			}

			// Hidden test hook: `/advisor test <nit|concern|blocker> <note>` drives the
			// real deliverAdvice routing without depending on the advisor model's
			// severity choice. Used by the RPC delivery tests.
			if (arg.startsWith("test")) {
				const parsed = parseAdvisorTestArgs(args);
				if (!parsed) {
					ctx.ui.notify("usage: /advisor test <nit|concern|blocker> <note>", "warning");
					return;
				}
				deliverAdvice(parsed.note, parsed.severity);
				return;
			}

			ctx.ui.notify("usage: /advisor [on|off|status]", "warning");
		},
	});
}
