/**
 * Tests for the /advisor extension (a persistent second model that reviews each
 * turn and injects advice). Mirrors review.test.mjs structure.
 *
 * Layers:
 *   1. pure logic        — severity helpers, backoff, terminal detection, arg
 *                          parsing, advisory/​delta formatting, AdviseTool dedup
 *                          (no model/network/TUI)
 *   1b. runtime mechanics — always-hold + catch-up block: runTurnBlock branches
 *                          (stub runtime) and the real AdvisorRuntime + stub
 *                          Agent (hold → reconfirm → deliver/drop, settle waits)
 *   2. real loader       — the extension registers through pi's loader
 *   3. render path        — the advisory renderer shows notes by severity
 *   4. pi harness (E2E)  — drive a real `pi --mode rpc` and verify a nit is
 *                          delivered immediately and triggers a turn. Gated
 *                          behind ADVISOR_E2E=1 (needs anthropic auth + network;
 *                          spawns pi with ADVISOR_NO_REVIEW so the advisor model
 *                          never fires — only the deterministic `/advisor test`
 *                          nit hook does; high-sev needs the runtime, covered in 1b).
 *
 * Run:  node packages/pi-omplike-advisor/extensions/advisor.test.mjs              (fast, offline)
 *       ADVISOR_E2E=1 node packages/pi-omplike-advisor/extensions/advisor.test.mjs (also the pi harness)
 */

import assert from "node:assert/strict";
import { spawn, execSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { StringDecoder } from "node:string_decoder";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const PI_BIN = execSync("command -v pi").toString().trim();
const DIST = dirname(execSync(`readlink -f ${PI_BIN}`).toString().trim());

const { createExtensionRuntime, loadExtensions } = await import(`${DIST}/core/extensions/loader.js`);
const { createEventBus } = await import(`${DIST}/core/event-bus.js`);
const { CustomMessageComponent } = await import(`${DIST}/modes/interactive/components/custom-message.js`);
const { initTheme } = await import(`${DIST}/modes/interactive/theme/theme.js`);

// advisor.ts has @earendil-works/* value imports; reach its exported pure helpers
// through jiti with the same aliases pi's extension loader uses.
const piRequire = createRequire(`${DIST}/index.js`);
const jitiDir = dirname(piRequire.resolve("jiti/package.json"));
const { createJiti } = await import(`${jitiDir}/lib/jiti-static.mjs`);
const pkgEntry = (pkg) => resolve(DIST, "..", "node_modules", "@earendil-works", pkg, "dist/index.js");
const ALIAS = {
	"@earendil-works/pi-coding-agent": `${DIST}/index.js`,
	"@earendil-works/pi-agent-core": pkgEntry("pi-agent-core"),
	"@earendil-works/pi-tui": pkgEntry("pi-tui"),
	"@earendil-works/pi-ai": pkgEntry("pi-ai"),
	typebox: resolve(DIST, "..", "node_modules", "typebox", "build", "index.mjs"),
};
const jiti = createJiti(import.meta.url, { moduleCache: false, alias: ALIAS });
const A = await jiti.import(resolve(HERE, "advisor.ts"));

initTheme();

let passed = 0;
const tests = [];
const test = (name, fn) => tests.push([name, fn]);
const strip = (s) => s.replace(/\x1b\[[0-9;]*m/g, "");

// ===========================================================================
// 1. pure logic
// ===========================================================================

test("isHighSeverity: only concern/blocker are held + reconfirmed", () => {
	assert.equal(A.isHighSeverity(undefined), false);
	assert.equal(A.isHighSeverity("nit"), false);
	assert.equal(A.isHighSeverity("concern"), true);
	assert.equal(A.isHighSeverity("blocker"), true);
});

test("nextBackoffMs: base, doubling, capped, guarded", () => {
	assert.equal(A.nextBackoffMs(0, 15000, 120000), 15000);
	assert.equal(A.nextBackoffMs(1, 15000, 120000), 30000);
	assert.equal(A.nextBackoffMs(2, 15000, 120000), 60000);
	assert.equal(A.nextBackoffMs(3, 15000, 120000), 120000);
	assert.equal(A.nextBackoffMs(4, 15000, 120000), 120000); // capped
	assert.equal(A.nextBackoffMs(-1, 15000, 120000), 15000); // negative guarded to base
	assert.equal(A.nextBackoffMs(0), 15000); // defaults
});

test("isTerminalTurn: terminal iff the assistant message made no tool calls", () => {
	assert.equal(A.isTerminalTurn({ content: [{ type: "text" }] }), true);
	assert.equal(A.isTerminalTurn({ content: [] }), true);
	assert.equal(A.isTerminalTurn(undefined), true);
	assert.equal(A.isTerminalTurn({ content: [{ type: "toolCall" }] }), false);
	assert.equal(A.isTerminalTurn({ content: [{ type: "text" }, { type: "toolCall" }] }), false);
});

test("formatReconfirmPreamble: empty when nothing held, else lists held notes", () => {
	assert.equal(A.formatReconfirmPreamble([]), "");
	const p = A.formatReconfirmPreamble([
		{ note: "races on shared map", severity: "blocker" },
		{ note: "missing await", severity: "concern" },
	]);
	assert.match(p, /Held advisories — reconfirm/);
	assert.match(p, /call `advise` again/);
	assert.match(p, /- \[BLOCKER\] races on shared map/);
	assert.match(p, /- \[CONCERN\] missing await/);
	assert.match(p, /\n---\n/); // separates preamble from the session update below
});

test("parseAdvisorTestArgs: valid severities + multiword note", () => {
	assert.deepEqual(A.parseAdvisorTestArgs("test nit be tidy"), { severity: "nit", note: "be tidy" });
	assert.deepEqual(A.parseAdvisorTestArgs("test  concern   wrong path here"), {
		severity: "concern",
		note: "wrong path here",
	});
	assert.deepEqual(A.parseAdvisorTestArgs("test BLOCKER STOP NOW"), { severity: "blocker", note: "STOP NOW" });
});

test("parseAdvisorTestArgs: rejects bad input", () => {
	assert.equal(A.parseAdvisorTestArgs("test"), null);
	assert.equal(A.parseAdvisorTestArgs("test nit"), null); // no note
	assert.equal(A.parseAdvisorTestArgs("test bogus hi"), null); // bad severity
	assert.equal(A.parseAdvisorTestArgs("status"), null);
});

test("formatAdvisoryContent: wraps with severity + guidance, escapes XML", () => {
	const c = A.formatAdvisoryContent([{ note: "use <T> & stuff", severity: "concern" }]);
	assert.match(c, /<advisory severity="concern" guidance="weigh, don't blindly obey">/);
	assert.match(c, /use &lt;T&gt; &amp; stuff/);
	assert.match(c, /<\/advisory>/);
});

test("formatAdvisoryContent: omits severity attr when absent (plain nit)", () => {
	const c = A.formatAdvisoryContent([{ note: "tidy up" }]);
	assert.doesNotMatch(c, /severity=/);
	assert.match(c, /<advisory guidance=/);
});

test("formatAdvisoryContent: stale option tags advice as about an earlier step", () => {
	const c = A.formatAdvisoryContent([{ note: "rename", severity: "nit" }], { stale: true });
	assert.match(c, /context="raised about an earlier step"/);
	assert.doesNotMatch(A.formatAdvisoryContent([{ note: "rename", severity: "nit" }]), /context=/);
});

test("formatTurnDelta: includes user, thinking, text, tool call + result", () => {
	const md = A.formatTurnDelta({
		userPrompt: "do the thing",
		assistant: {
			role: "assistant",
			content: [
				{ type: "thinking", thinking: "let me think" },
				{ type: "text", text: "here is my plan" },
				{ type: "toolCall", id: "1", name: "write", arguments: { path: "a.js" } },
			],
			usage: {},
			stopReason: "toolUse",
			timestamp: 1,
		},
		toolResults: [{ role: "toolResult", toolCallId: "1", toolName: "write", content: [{ type: "text", text: "wrote a.js" }], isError: false, timestamp: 2 }],
	});
	assert.match(md, /#### User\n\ndo the thing/);
	assert.match(md, /<thinking>\nlet me think\n<\/thinking>/);
	assert.match(md, /here is my plan/);
	assert.match(md, /→ tool `write`\(\{"path":"a\.js"\}\)/);
	assert.match(md, /#### Tool result: `write`\n\nwrote a\.js/);
});

test("formatTurnDelta: edits render as compact header + result diff (no raw old/new blobs)", () => {
	const diff = "  10 unchanged\n- 11 bootstrap 0/0\n+ 11 bootstrap 0.045% (9/20000)\n  12 unchanged";
	const md = A.formatTurnDelta({
		assistant: {
			role: "assistant",
			content: [
				{
					type: "toolCall",
					id: "1",
					name: "edit",
					arguments: {
						path: "RESULTS.md",
						edits: [
							{ oldText: "bootstrap 0/0", newText: "bootstrap 0.045% (9/20000)" },
							{ oldText: "x", newText: "y" },
						],
					},
				},
			],
			usage: {},
			stopReason: "toolUse",
			timestamp: 1,
		},
		toolResults: [
			{
				role: "toolResult",
				toolCallId: "1",
				toolName: "edit",
				content: [{ type: "text", text: "Successfully replaced 2 block(s)." }],
				details: { diff },
				isError: false,
				timestamp: 2,
			},
		],
	});
	// compact toolCall header, not the raw {oldText,newText} JSON dump
	assert.ok(md.includes("→ tool `edit`(RESULTS.md) — 2 block(s); diff in tool result"));
	// the result body is the marked diff (with -/+ framing), not the success text
	assert.ok(md.includes("- 11 bootstrap 0/0"));
	assert.ok(md.includes("+ 11 bootstrap 0.045% (9/20000)"));
	// the stale pre-edit blob must NOT appear as an unannotated peer (only inside the diff, prefixed)
	assert.ok(!md.includes('"oldText"'));
	assert.ok(!md.includes("Successfully replaced"));
});

test("formatTurnDelta: a failed edit (no diff) keeps its attempted args for diagnosis", () => {
	const md = A.formatTurnDelta({
		assistant: {
			role: "assistant",
			content: [
				{ type: "toolCall", id: "9", name: "edit", arguments: { path: "f.py", edits: [{ oldText: "needle that did not match", newText: "x" }] } },
			],
			usage: {},
			stopReason: "toolUse",
			timestamp: 1,
		},
		// failed edit: error result, NO details.diff
		toolResults: [
			{ role: "toolResult", toolCallId: "9", toolName: "edit", content: [{ type: "text", text: "Error: oldText not found" }], isError: true, timestamp: 2 },
		],
	});
	assert.ok(md.includes("needle that did not match"), "attempted oldText must survive when there is no diff");
	assert.ok(!md.includes("diff in tool result"), "no compact/diff header without a diff");
	assert.ok(md.includes("`edit` (error)"), "the error is still shown");
});

test("formatTurnDelta: an ERROR result with a diff keeps args + error text (untrusted diff dropped)", () => {
	const md = A.formatTurnDelta({
		assistant: {
			role: "assistant",
			content: [
				{ type: "toolCall", id: "7", name: "multiedit", arguments: { path: "g.py", edits: [{ oldText: "attempted needle", newText: "z" }] } },
			],
			usage: {},
			stopReason: "toolUse",
			timestamp: 1,
		},
		// a custom/hooked edit tool that errored but still carried a diff
		toolResults: [
			{
				role: "toolResult",
				toolCallId: "7",
				toolName: "multiedit",
				content: [{ type: "text", text: "Error: partial apply rejected" }],
				details: { diff: "- 1 old\n+ 1 new" },
				isError: true,
				timestamp: 2,
			},
		],
	});
	assert.ok(md.includes("attempted needle"), "args kept on error even when a diff exists");
	assert.ok(md.includes("Error: partial apply rejected"), "error text shown, not replaced by the diff");
	assert.ok(!md.includes("+ 1 new"), "untrustworthy error-result diff is not shown");
	assert.ok(!md.includes("diff in tool result"), "no suppression header on an error result");
});

test("formatTurnDelta: feeds large content verbatim (no truncation, no markers)", () => {
	const big = "LINE\n".repeat(5000); // ~25KB, well past every old clamp
	const md = A.formatTurnDelta({
		userPrompt: big,
		assistant: {
			role: "assistant",
			content: [{ type: "text", text: big }],
			usage: {},
			stopReason: "toolUse",
			timestamp: 1,
		},
		toolResults: [{ role: "toolResult", toolCallId: "1", toolName: "bash", content: [{ type: "text", text: big }], isError: false, timestamp: 2 }],
	});
	assert.ok(!md.includes("truncated"), "nothing should be truncated");
	assert.ok(md.includes(big), "content rides verbatim");
});

test("formatTurnDelta: marks tool errors", () => {
	const md = A.formatTurnDelta({
		toolResults: [{ role: "toolResult", toolCallId: "1", toolName: "bash", content: [{ type: "text", text: "boom" }], isError: true, timestamp: 2 }],
	});
	assert.match(md, /#### Tool result: `bash` \(error\)/);
});

test("formatTurnDelta: empty turn ⇒ empty string", () => {
	assert.equal(A.formatTurnDelta({}), "");
});

test("AdviseTool: records, dedups, and escalates by severity rank", async () => {
	const calls = [];
	const tool = new A.AdviseTool((note, severity) => calls.push({ note, severity }));

	const r1 = await tool.execute("c1", { note: "guard empty array", severity: "nit" });
	assert.equal(calls.length, 1);
	assert.match(r1.content[0].text, /Recorded/);

	// exact duplicate (same text, same severity) is dropped
	const r2 = await tool.execute("c2", { note: "guard empty array", severity: "nit" });
	assert.equal(calls.length, 1);
	assert.match(r2.content[0].text, /Duplicate/);

	// whitespace-normalized duplicate also dropped
	await tool.execute("c3", { note: "guard   empty\narray", severity: "nit" });
	assert.equal(calls.length, 1);

	// escalation to a higher severity passes through
	await tool.execute("c4", { note: "guard empty array", severity: "concern" });
	assert.equal(calls.length, 2);
	assert.equal(calls[1].severity, "concern");

	// de-escalation back down is dropped
	await tool.execute("c5", { note: "guard empty array", severity: "nit" });
	assert.equal(calls.length, 2);

	// reset clears memory ⇒ same note can be raised again
	tool.resetDelivered();
	await tool.execute("c6", { note: "guard empty array", severity: "nit" });
	assert.equal(calls.length, 3);
});

test("AdviseTool: held notes (onAdvice→false) stay unrecorded so they can re-fire", async () => {
	let deliver = false; // simulate "held" first, then "delivered"
	const calls = [];
	const tool = new A.AdviseTool((note, severity) => {
		calls.push({ note, severity });
		return deliver;
	});

	// first attempt held → tool reports held, dedup NOT recorded
	const r1 = await tool.execute("h1", { note: "data race", severity: "blocker" });
	assert.match(r1.content[0].text, /Held/);
	assert.equal(r1.details.held, true);
	assert.equal(calls.length, 1);

	// same note re-raised while still held → onAdvice fires AGAIN (not deduped away)
	await tool.execute("h2", { note: "data race", severity: "blocker" });
	assert.equal(calls.length, 2);

	// now it gets delivered → recorded
	deliver = true;
	const r3 = await tool.execute("h3", { note: "data race", severity: "blocker" });
	assert.match(r3.content[0].text, /Recorded/);
	assert.equal(calls.length, 3);

	// once delivered, a same-severity repeat is deduped away
	await tool.execute("h4", { note: "data race", severity: "blocker" });
	assert.equal(calls.length, 3);
});

test("AdviseTool: markDelivered records dedup at the real delivery point", async () => {
	const calls = [];
	const tool = new A.AdviseTool((note, severity) => {
		calls.push({ note, severity });
		return false; // always held (high-severity path)
	});
	// the catch-up block delivers a held note, then records it:
	tool.markDelivered("data race", "blocker");
	// a later same-severity re-raise is now deduped before onAdvice fires
	const r = await tool.execute("x", { note: "data race", severity: "blocker" });
	assert.match(r.content[0].text, /Duplicate/);
	assert.equal(calls.length, 0);
	// but a genuine escalation past the recorded rank still passes
	const tool2 = new A.AdviseTool((note, severity) => {
		calls.push({ note, severity });
		return false;
	});
	tool2.markDelivered("flaky", "concern");
	await tool2.execute("y", { note: "flaky", severity: "blocker" });
	assert.equal(calls.length, 1);
});

// ===========================================================================
// 1b. runtime mechanics (offline, stub agent) — always-hold + catch-up block
//
// The hold/reconfirm/deliver flow needs the real runtime + a controllable
// advisor, which a live E2E can't make deterministic (the /advisor test hook
// bypasses the runtime entirely). So we drive runTurnBlock with a stub runtime,
// and the real AdvisorRuntime with a stub Agent.
// ===========================================================================

// --- runTurnBlock orchestration, against a stub runtime ---
function stubRuntime({ held = [], settleResult = "settled" } = {}) {
	return {
		_held: [...held],
		waited: false,
		get hasHeld() {
			return this._held.length > 0;
		},
		takeHeld() {
			return this._held.splice(0);
		},
		async waitUntilSettled() {
			this.waited = true;
			return settleResult;
		},
	};
}
const blockArgs = (over) => ({ consecutiveBlocks: 0, notify: () => {}, deliverHeld: () => {}, ...over });

test("runTurnBlock: non-terminal with nothing held → no block, streak resets", async () => {
	const rt = stubRuntime({ held: [] });
	const delivered = [];
	const n = await A.runTurnBlock(blockArgs({ terminal: false, runtime: rt, consecutiveBlocks: 3, deliverHeld: (x) => delivered.push(...x) }));
	assert.equal(n, 0);
	assert.equal(rt.waited, false, "must not block");
	assert.equal(delivered.length, 0);
});

test("runTurnBlock: non-terminal + held + settled → delivers survivors, resets streak", async () => {
	const delivered = [];
	const rt = stubRuntime({ held: [{ note: "x", severity: "blocker" }], settleResult: "settled" });
	const n = await A.runTurnBlock(blockArgs({ terminal: false, runtime: rt, consecutiveBlocks: 2, deliverHeld: (x) => delivered.push(...x) }));
	assert.equal(n, 0);
	assert.deepEqual(delivered, [{ note: "x", severity: "blocker" }]);
});

test("runTurnBlock: non-terminal + held + timeout → keeps held, doubles streak", async () => {
	const delivered = [];
	const rt = stubRuntime({ held: [{ note: "x", severity: "blocker" }], settleResult: "timeout" });
	const n = await A.runTurnBlock(blockArgs({ terminal: false, runtime: rt, consecutiveBlocks: 1, deliverHeld: (x) => delivered.push(...x) }));
	assert.equal(n, 2, "streak doubles via consecutiveBlocks+1");
	assert.equal(delivered.length, 0);
	assert.equal(rt.hasHeld, true, "held notes are kept, not taken");
});

test("runTurnBlock: terminal blocks unconditionally (even with nothing held)", async () => {
	const rt = stubRuntime({ held: [], settleResult: "settled" });
	const n = await A.runTurnBlock(blockArgs({ terminal: true, runtime: rt }));
	assert.equal(rt.waited, true, "terminal must block until the advisor settles");
	assert.equal(n, 0);
});

test("runTurnBlock: terminal timeout → delivers held best-effort (current, not stale)", async () => {
	const delivered = [];
	const rt = stubRuntime({ held: [{ note: "x", severity: "concern" }], settleResult: "timeout" });
	const n = await A.runTurnBlock(blockArgs({ terminal: true, runtime: rt, deliverHeld: (x) => delivered.push(...x) }));
	assert.equal(n, 0);
	assert.deepEqual(delivered, [{ note: "x", severity: "concern" }]);
});

test("runTurnBlock: aborted (user Escape) → keeps held + streak, no delivery", async () => {
	const delivered = [];
	const rt = stubRuntime({ held: [{ note: "x", severity: "blocker" }], settleResult: "aborted" });
	const n = await A.runTurnBlock(blockArgs({ terminal: false, runtime: rt, consecutiveBlocks: 2, deliverHeld: (x) => delivered.push(...x) }));
	assert.equal(n, 2, "streak preserved");
	assert.equal(delivered.length, 0);
	assert.equal(rt.hasHeld, true);
});

test("runTurnBlock: non-terminal + failed reconfirm → keeps held unconfirmed, backs off", async () => {
	// A failed reconfirm (advisor errored out) must NOT deliver held notes as if
	// confirmed — same handling as a timeout.
	const delivered = [];
	const rt = stubRuntime({ held: [{ note: "x", severity: "blocker" }], settleResult: "failed" });
	const n = await A.runTurnBlock(blockArgs({ terminal: false, runtime: rt, consecutiveBlocks: 1, deliverHeld: (x) => delivered.push(...x) }));
	assert.equal(n, 2, "backoff lengthens");
	assert.equal(delivered.length, 0, "unconfirmed held note is NOT delivered mid-run");
	assert.equal(rt.hasHeld, true);
});

test("runTurnBlock: terminal + failed reconfirm → best-effort delivers", async () => {
	const delivered = [];
	const rt = stubRuntime({ held: [{ note: "x", severity: "concern" }], settleResult: "failed" });
	const n = await A.runTurnBlock(blockArgs({ terminal: true, runtime: rt, deliverHeld: (x) => delivered.push(...x) }));
	assert.equal(n, 0);
	assert.deepEqual(delivered, [{ note: "x", severity: "concern" }], "last chance before idle → deliver best-effort");
});

// --- real AdvisorRuntime + stub Agent: hold → reconfirm → deliver/drop ---
// onReview(text, {tool, rt, reviewCount}) simulates the advisor's reaction per review.
function buildIntegration({ onReview } = {}) {
	const delivered = [];
	let rt;
	let reviewCount = 0;
	const tool = new A.AdviseTool((note, severity) => {
		// mirrors deliverAdvice: drop stale (orphaned review), hold high severity,
		// treat a nit matching a held note as a reconfirmation, else deliver the nit.
		if (rt && !rt.acceptingAdvice) return true;
		if (A.isHighSeverity(severity)) {
			rt.hold(note, severity);
			return false;
		}
		if (rt.reconfirmIfHeld(note)) return true;
		delivered.push({ note, severity, kind: "nit" });
		return true;
	});
	const agent = {
		state: { messages: [], model: {} },
		async prompt(text) {
			// Defer like a real (multi-second, network) advisor review: the hold must
			// land AFTER push()/turn_end returns, not synchronously inside it.
			await new Promise((r) => setTimeout(r, 0));
			reviewCount++;
			await onReview?.(text, { tool, rt, reviewCount });
			this.state.messages.push({ role: "assistant", content: [], usage: {}, stopReason: "stop" });
		},
		abort() {},
		reset() {
			this.state.messages = [];
		},
	};
	rt = new A.AdvisorRuntime(agent, tool, 0);
	// mirrors the extension's deliverHeld: steer survivors + record dedup
	const deliverHeld = (notes) => {
		for (const n of notes) {
			delivered.push({ ...n, kind: "held" });
			tool.markDelivered(n.note, n.severity);
		}
	};
	const block = (terminal, opts = {}) =>
		A.runTurnBlock({ terminal, runtime: rt, consecutiveBlocks: 0, notify: () => {}, deliverHeld, ...opts });
	return { rt, tool, delivered, deliverHeld, block, getReviewCount: () => reviewCount };
}

test("integration: a nit is delivered during review, not held, never blocks", async () => {
	const h = buildIntegration({
		onReview: async (_t, { tool, reviewCount }) => {
			if (reviewCount === 1) await tool.execute("n1", { note: "rename var", severity: "nit" });
		},
	});
	h.rt.push("turn 1");
	const cb = await h.block(false);
	await h.rt.waitUntilSettled(5000);
	assert.equal(cb, 0, "no block (nits never hold)");
	assert.equal(h.rt.hasHeld, false);
	assert.equal(h.delivered.length, 1);
	assert.equal(h.delivered[0].kind, "nit");
});

test("integration: blocker held on turn 1, survives reconfirm, delivered after terminal block", async () => {
	const h = buildIntegration({
		onReview: async (text, { tool, reviewCount }) => {
			if (reviewCount === 1) {
				await tool.execute("a1", { note: "off-by-one", severity: "blocker" });
			} else if (reviewCount === 2) {
				assert.match(text, /Held advisories/, "reconfirm preamble rides review 2");
				await tool.execute("a2", { note: "off-by-one", severity: "blocker" }); // still applies
			}
		},
	});
	// turn 1: non-terminal, nothing held yet → no block; review 1 holds the blocker
	h.rt.push("turn 1");
	assert.equal(await h.block(false), 0);
	await h.rt.waitUntilSettled(5000);
	assert.equal(h.rt.hasHeld, true);
	assert.equal(h.delivered.length, 0, "nothing delivered on the flagging turn");
	// turn 2: terminal → block until settled; review 2 reconfirms; survivor delivered
	h.rt.push("turn 2");
	assert.equal(await h.block(true), 0);
	assert.equal(h.getReviewCount(), 2);
	assert.equal(h.delivered.length, 1);
	assert.equal(h.delivered[0].kind, "held");
	assert.equal(h.delivered[0].severity, "blocker");
});

test("integration: a blocker first raised ON the terminal turn is caught + delivered (Q1)", async () => {
	// The advisor flags for the first time while the terminal turn is blocked; the
	// agent did no follow-up (it's stopping), so it's delivered without a reconfirm.
	const h = buildIntegration({
		onReview: async (_text, { tool, reviewCount }) => {
			if (reviewCount === 1) await tool.execute("a1", { note: "leaks an fd", severity: "blocker" });
		},
	});
	h.rt.push("final turn");
	assert.equal(await h.block(true), 0, "terminal block waits for the review that raises the blocker");
	assert.equal(h.getReviewCount(), 1);
	assert.equal(h.delivered.length, 1, "blocker raised on the terminal turn lands before idle");
	assert.equal(h.delivered[0].kind, "held");
	assert.equal(h.delivered[0].severity, "blocker");
	assert.equal(h.rt.hasHeld, false);
});

test("integration (F1): advice from a review orphaned by reset() is dropped, not held", async () => {
	const h = buildIntegration({
		onReview: async (_t, { tool, rt, reviewCount }) => {
			if (reviewCount === 1) {
				rt.reset(); // orphan this review mid-flight (e.g. session compaction)
				await tool.execute("a1", { note: "stale blocker", severity: "blocker" });
			}
		},
	});
	h.rt.push("turn 1");
	await h.block(false);
	await h.rt.waitUntilSettled(2000);
	assert.equal(h.rt.hasHeld, false, "orphaned review's hold is dropped");
	assert.equal(h.delivered.length, 0, "nothing delivered from a stale review");
});

test("integration (F2): a held blocker re-raised as a nit is kept, not de-escalated", async () => {
	const h = buildIntegration({
		onReview: async (_text, { tool, reviewCount }) => {
			if (reviewCount === 1) await tool.execute("a1", { note: "off-by-one", severity: "blocker" });
			else if (reviewCount === 2) await tool.execute("a2", { note: "off-by-one", severity: "nit" }); // de-escalation attempt
		},
	});
	h.rt.push("turn 1");
	await h.block(false);
	await h.rt.waitUntilSettled(5000);
	assert.equal(h.rt.hasHeld, true);
	h.rt.push("turn 2");
	assert.equal(await h.block(true), 0);
	assert.equal(h.delivered.length, 1, "no nit delivered; the held note survives");
	assert.equal(h.delivered[0].kind, "held");
	assert.equal(h.delivered[0].severity, "blocker", "kept at blocker severity, not lowered to nit");
});

test("integration: held blocker is dropped when the reconfirm review recants", async () => {
	const h = buildIntegration({
		onReview: async (_text, { tool, reviewCount }) => {
			if (reviewCount === 1) await tool.execute("a1", { note: "off-by-one", severity: "blocker" });
			// review 2: agent fixed it → advisor stays silent → held note evaporates
		},
	});
	h.rt.push("turn 1");
	await h.block(false);
	await h.rt.waitUntilSettled(5000);
	assert.equal(h.rt.hasHeld, true);
	h.rt.push("turn 2");
	assert.equal(await h.block(true), 0);
	assert.equal(h.delivered.length, 0, "recanted blocker is dropped, not delivered");
	assert.equal(h.rt.hasHeld, false);
});

test("integration (regression): a held note survives push() and blocks + delivers mid-run", async () => {
	// Regression for the synchronous-#drain-splice bug: push() runs the drain up to
	// its first await, which must NOT empty #held — otherwise a non-terminal turn
	// sees hasHeld=false and never blocks, deferring high-sev delivery to terminal.
	const h = buildIntegration({
		onReview: async (text, { tool, reviewCount }) => {
			if (reviewCount === 1) await tool.execute("a1", { note: "races on cache", severity: "blocker" });
			else if (reviewCount === 2) {
				assert.match(text, /Held advisories/);
				await tool.execute("a2", { note: "races on cache", severity: "blocker" }); // still applies
			}
		},
	});
	h.rt.push("turn 1");
	await h.block(false);
	await h.rt.waitUntilSettled(5000);
	assert.equal(h.rt.hasHeld, true);
	// turn 2 is NON-terminal; the held note must keep hasHeld true across push
	h.rt.push("turn 2");
	assert.equal(h.rt.hasHeld, true, "held note survives push() (no mid-flight splice)");
	const cb = await h.block(false);
	assert.equal(h.delivered.length, 1, "prior held blocker delivered mid-run, not deferred to terminal");
	assert.equal(h.delivered[0].kind, "held");
	assert.equal(cb, 0, "settled → streak reset");
});

test("integration (regression): terminal timeout delivers a held note stuck mid-reconfirm", async () => {
	// Regression for Finding 2: a pre-existing held note must remain in #held while
	// its reconfirm review is in flight, so a terminal timeout can still deliver it.
	let releaseReview2;
	const h = buildIntegration({
		onReview: async (_text, { tool, reviewCount }) => {
			if (reviewCount === 1) await tool.execute("a1", { note: "fd leak", severity: "blocker" });
			else if (reviewCount === 2) await new Promise((r) => (releaseReview2 = r)); // hang past the timeout
		},
	});
	h.rt.push("turn 1");
	await h.block(false);
	await h.rt.waitUntilSettled(5000);
	assert.equal(h.rt.hasHeld, true);
	h.rt.push("turn 2");
	const cb = await h.block(true, { capMs: 30 }); // terminal, review 2 hangs → times out
	assert.equal(h.delivered.length, 1, "pre-existing held note delivered best-effort on terminal timeout");
	assert.equal(h.delivered[0].severity, "blocker");
	assert.equal(cb, 0);
	releaseReview2?.(); // let the hung review finish for a clean exit
});

test("runtime.waitUntilSettled: settles on drain, times out, and aborts", async () => {
	let resolvePrompt;
	const agent = {
		state: { messages: [], model: {} },
		prompt() {
			return new Promise((r) => {
				resolvePrompt = r;
			});
		},
		abort() {},
		reset() {},
	};
	const rt = new A.AdvisorRuntime(agent, new A.AdviseTool(() => true), 0);
	rt.push("hang"); // drain starts, prompt hangs → not idle
	assert.equal(await rt.waitUntilSettled(20), "timeout");
	const ac = new AbortController();
	const p = rt.waitUntilSettled(2000, ac.signal);
	ac.abort();
	assert.equal(await p, "aborted");
	resolvePrompt(); // let the drain finish
	assert.equal(await rt.waitUntilSettled(2000), "settled");
});

test("runtime.waitUntilSettled: a dropped (3x-failed) review resolves 'failed', held preserved", async () => {
	let attempts = 0;
	const agent = {
		state: { messages: [], model: {} },
		async prompt() {
			attempts++;
			throw new Error("boom");
		},
		abort() {},
		reset() {},
	};
	const rt = new A.AdvisorRuntime(agent, new A.AdviseTool(() => false), 0);
	rt.hold("data race", "blocker"); // pre-existing held note
	rt.push("turn");
	assert.equal(await rt.waitUntilSettled(2000), "failed");
	assert.equal(attempts, 3, "retried 3x then dropped");
	assert.equal(rt.hasHeld, true, "held note preserved across a failed reconfirm");
});

test("runtime.waitUntilSettled: a provider error (stopReason, no throw) resolves 'failed', held preserved", async () => {
	// The real Agent records provider failures as an assistant message with
	// stopReason "error" rather than throwing — that must count as a failed review.
	let attempts = 0;
	const agent = {
		state: { messages: [], model: {} },
		async prompt() {
			attempts++;
			this.state.messages.push({ role: "assistant", content: [], usage: {}, stopReason: "error", errorMessage: "503" });
		},
		abort() {},
		reset() {},
	};
	const rt = new A.AdvisorRuntime(agent, new A.AdviseTool(() => false), 0);
	rt.hold("data race", "blocker");
	rt.push("turn");
	assert.equal(await rt.waitUntilSettled(2000), "failed");
	assert.equal(attempts, 3, "errored review retried 3x then dropped");
	assert.equal(rt.hasHeld, true, "held note NOT pruned by an errored (non-throwing) review");
});

test("runtime.waitUntilSettled: reset() cancels a pending waiter as 'aborted' immediately", async () => {
	let resolvePrompt;
	const agent = {
		state: { messages: [], model: {} },
		prompt() {
			return new Promise((r) => (resolvePrompt = r)); // hang
		},
		abort() {},
		reset() {},
	};
	const rt = new A.AdvisorRuntime(agent, new A.AdviseTool(() => true), 0);
	rt.push("hang");
	const p = rt.waitUntilSettled(5000); // would hang on the in-flight prompt
	rt.reset(); // must resolve the waiter now, not wait for the prompt/timeout
	assert.equal(await p, "aborted");
	resolvePrompt?.(); // let the hung prompt unwind for a clean exit
});

test("runtime.waitUntilSettled: a batch that overflows even a FRESH context self-compacts once, then resolves 'failed', held preserved", async () => {
	// stopReason "length" triggers a one-shot reactive self-compaction (clear the
	// advisor's own history + replay the batch fresh). If the FRESH replay still
	// overflows, the batch genuinely doesn't fit, so it counts as a failed review
	// (retried up to 3x). Each outer review attempt therefore issues 2 prompts
	// (initial + one fresh-replay) = 6 total across the 3 retries.
	let attempts = 0;
	let resets = 0;
	const agent = {
		state: { messages: [], model: {} },
		async prompt() {
			attempts++;
			this.state.messages.push({ role: "assistant", content: [], usage: {}, stopReason: "length" });
		},
		abort() {},
		reset() {
			resets++;
			this.state.messages = [];
		},
	};
	const rt = new A.AdvisorRuntime(agent, new A.AdviseTool(() => false), 0);
	rt.hold("data race", "blocker");
	rt.push("turn");
	assert.equal(await rt.waitUntilSettled(2000), "failed");
	assert.equal(attempts, 6, "each of the 3 review retries self-compacts once then re-overflows (2 prompts each)");
	assert.equal(resets, 3, "one reactive self-compaction per review retry");
	assert.equal(rt.hasHeld, true, "held note NOT pruned by a truncated review");
});

test("runtime.waitUntilSettled: an ACCUMULATED-context overflow self-compacts and the fresh replay succeeds", async () => {
	// The common case: the advisor's own accumulated transcript overflowed, but the
	// batch fits fine in a fresh context. One reactive self-compaction recovers it —
	// the review then succeeds (settled, not failed) and recanted holds are pruned.
	let attempts = 0;
	let resets = 0;
	const agent = {
		state: { messages: [{ role: "user", content: [] }, { role: "assistant", content: [] }], model: {} },
		async prompt() {
			attempts++;
			// First attempt overflows (accumulated context); after a self-compaction the
			// fresh replay (empty messages) succeeds.
			const overflow = this.state.messages.length > 0;
			this.state.messages.push({ role: "assistant", content: [], usage: {}, stopReason: overflow ? "length" : "stop" });
		},
		abort() {},
		reset() {
			resets++;
			this.state.messages = [];
		},
	};
	const rt = new A.AdvisorRuntime(agent, new A.AdviseTool(() => false), 0);
	rt.hold("data race", "blocker"); // offered as preamble; advisor stays silent → pruned on success
	rt.push("turn");
	assert.equal(await rt.waitUntilSettled(2000), "settled");
	assert.equal(attempts, 2, "overflow then one successful fresh replay");
	assert.equal(resets, 1, "exactly one reactive self-compaction");
	assert.equal(rt.hasHeld, false, "a successful (post-compaction) review still prunes recanted holds");
});

test("runtime: a concern/blocker held by a DISCARDED overflowed attempt is rolled back, not kept (finding #1)", async () => {
	// Attempt 1 holds a blocker, then overflows (stopReason length). The reactive
	// self-compaction must roll that hold back: it was raised against a truncated
	// view, and offeredKeys (snapshotted pre-attempt) can't prune it. The fresh
	// replay re-raises only what still applies; here it stays silent, so the
	// phantom blocker must NOT survive (else it'd later deliver as if confirmed).
	let attempts = 0;
	let rt;
	const agent = {
		state: { messages: [], model: {} },
		async prompt() {
			attempts++;
			if (attempts === 1) {
				rt.hold("phantom blocker from overflowed attempt", "blocker"); // mid-attempt hold
				this.state.messages.push({ role: "assistant", content: [], usage: {}, stopReason: "length" });
			} else {
				this.state.messages.push({ role: "assistant", content: [], usage: {}, stopReason: "stop" }); // silent fresh replay
			}
		},
		abort() {},
		reset() {
			this.state.messages = [];
		},
	};
	rt = new A.AdvisorRuntime(agent, new A.AdviseTool(() => false), 0);
	rt.push("turn");
	assert.equal(await rt.waitUntilSettled(2000), "settled");
	assert.equal(attempts, 2, "overflow then a successful fresh replay");
	assert.equal(rt.hasHeld, false, "the phantom blocker from the discarded attempt was rolled back");
});

test("runtime: the reactive rollback keeps PRE-EXISTING held notes, dropping only the discarded attempt's adds (finding #1)", async () => {
	let attempts = 0;
	let rt;
	const agent = {
		state: { messages: [], model: {} },
		async prompt() {
			attempts++;
			if (attempts === 1) rt.hold("phantom from overflowed attempt", "blocker"); // only the first attempt adds it
			this.state.messages.push({ role: "assistant", content: [], usage: {}, stopReason: "length" }); // always overflows ⇒ failed, no prune
		},
		abort() {},
		reset() {
			this.state.messages = [];
		},
	};
	rt = new A.AdvisorRuntime(agent, new A.AdviseTool(() => false), 0);
	rt.hold("real prior blocker", "blocker"); // pre-existing, captured in the pre-batch snapshot
	rt.push("turn");
	assert.equal(await rt.waitUntilSettled(2000), "failed");
	const held = rt.takeHeld();
	assert.equal(held.length, 1, "exactly the pre-existing held note remains");
	assert.equal(held[0].note, "real prior blocker", "phantom dropped by rollback, prior kept");
});

test("runtime: PROACTIVE self-compaction fires at ADVISOR_COMPACT_AT, replays fresh, and preserves lifetime cost accounting", async () => {
	const promptMsgCounts = [];
	let resets = 0;
	const agent = {
		state: {
			// One prior turn whose usage puts the advisor at 90% of a 100k window.
			messages: [{ role: "assistant", content: [], usage: { input: 90000, cost: { total: 0.5 } }, stopReason: "stop" }],
			model: { contextWindow: 100000 },
		},
		async prompt() {
			promptMsgCounts.push(this.state.messages.length); // 0 ⇒ replayed into a fresh context
			this.state.messages.push({ role: "assistant", content: [], usage: { input: 5, cost: { total: 0.01 } }, stopReason: "stop" });
		},
		abort() {},
		reset() {
			resets++;
			this.state.messages = [];
		},
	};
	const rt = new A.AdvisorRuntime(agent, new A.AdviseTool(() => false), 0); // default compactAt = 80
	rt.push("turn");
	assert.equal(await rt.waitUntilSettled(2000), "settled");
	assert.equal(resets, 1, "proactive self-compaction reset the agent before the review");
	assert.equal(promptMsgCounts[0], 0, "the batch was replayed into a fresh (cleared) context");
	const u = rt.usage;
	assert.equal(u.input, 90005, "lifetime input survives the self-compaction (folded 90000 + fresh 5)");
	assert.ok(Math.abs(u.cost - 0.51) < 1e-9, "lifetime cost survives the self-compaction (0.5 + 0.01)");
	assert.equal(u.contextTokens, 5, "context size reflects only the fresh post-compaction turn");
});

test("runtime: PROACTIVE self-compaction respects a custom compactAtPercent threshold (no reset below it)", async () => {
	let resets = 0;
	const agent = {
		state: {
			messages: [{ role: "assistant", content: [], usage: { input: 60000 }, stopReason: "stop" }], // 60% of 100k
			model: { contextWindow: 100000 },
		},
		async prompt() {
			this.state.messages.push({ role: "assistant", content: [], usage: { input: 5 }, stopReason: "stop" });
		},
		abort() {},
		reset() {
			resets++;
			this.state.messages = [];
		},
	};
	// Threshold 95 > 60% current ⇒ must NOT compact.
	const rt = new A.AdvisorRuntime(agent, new A.AdviseTool(() => false), 0, undefined, 95);
	rt.push("turn");
	assert.equal(await rt.waitUntilSettled(2000), "settled");
	assert.equal(resets, 0, "context below the configured threshold is left intact");
});

test("runtime.acceptingAdvice: an in-flight review orphaned by reset() stops accepting advice", async () => {
	let during;
	let afterReset;
	let rt;
	const agent = {
		state: { messages: [], model: {} },
		async prompt() {
			during = rt.acceptingAdvice; // reviewEpoch === epoch
			rt.reset(); // bumps epoch → orphans this in-flight review
			afterReset = rt.acceptingAdvice;
			this.state.messages.push({ role: "assistant", content: [], usage: {}, stopReason: "stop" });
		},
		abort() {},
		reset() {},
	};
	rt = new A.AdvisorRuntime(agent, new A.AdviseTool(() => false), 0);
	rt.push("turn");
	await rt.waitUntilSettled(2000);
	assert.equal(during, true, "advice accepted during a live review");
	assert.equal(afterReset, false, "advice rejected once the review's epoch is orphaned");
});

test("runtime.hold: re-raising a held note at higher severity escalates it", () => {
	const rt = new A.AdvisorRuntime({ state: { messages: [], model: {} }, async prompt() {}, abort() {}, reset() {} }, new A.AdviseTool(() => false), 0);
	rt.hold("flaky test", "concern");
	rt.hold("flaky   test", "blocker"); // same note (whitespace-normalized), escalated
	const held = rt.takeHeld();
	assert.equal(held.length, 1, "deduped to one entry");
	assert.equal(held[0].severity, "blocker", "escalation honored");
	// de-escalation is ignored
	rt.hold("x", "blocker");
	rt.hold("x", "concern");
	assert.equal(rt.takeHeld()[0].severity, "blocker");
});

// ===========================================================================
// 2. real loader
// ===========================================================================

async function loadAdvisorExtension() {
	const runtime = createExtensionRuntime();
	const res = await loadExtensions(["advisor.ts"], HERE, createEventBus(), runtime);
	assert.deepEqual(res.errors, [], "extension should load without errors");
	return res.extensions[0];
}

test("extension loads + registers /advisor command and advisory renderer", async () => {
	const ext = await loadAdvisorExtension();
	assert.ok(ext.commands.has("advisor"), "registers /advisor");
	assert.ok(ext.messageRenderers.has("advisory"), "registers advisory renderer");
});

// ===========================================================================
// 3. render path
// ===========================================================================

async function renderAdvisory(notes) {
	const ext = await loadAdvisorExtension();
	const renderer = ext.messageRenderers.get("advisory");
	const message = {
		role: "custom",
		customType: "advisory",
		content: [{ type: "text", text: "x" }],
		display: true,
		details: { notes },
		timestamp: Date.now(),
	};
	const comp = new CustomMessageComponent(message, renderer);
	comp.setExpanded(false);
	return strip(comp.render(100).join("\n"));
}

test("render: advisory card shows severity tag + note text", async () => {
	const text = await renderAdvisory([{ note: "this divides by zero on empty input", severity: "blocker" }]);
	assert.match(text, /advisor/i);
	assert.match(text, /BLOCKER/);
	assert.match(text, /divides by zero/);
});

test("render: plain nit shows NIT tag", async () => {
	const text = await renderAdvisory([{ note: "tidy this up" }]);
	assert.match(text, /NIT/);
	assert.match(text, /tidy this up/);
});

// ===========================================================================
// 4. pi harness (E2E) — nit delivers immediately + triggers a turn
//
// Only the nit path is live-testable: the /advisor test hook runs under
// ADVISOR_NO_REVIEW (no advisor model), so high-severity notes have no runtime
// to hold them and no turn_end block to deliver them. The hold → reconfirm →
// catch-up-block → deliver flow is covered deterministically by the offline
// runtime tests above.
// ===========================================================================

class RpcPi {
	constructor() {
		const cwd = mkdtempSync(join(tmpdir(), "advisor-e2e-"));
		execSync("git init -q", { cwd });
		writeFileSync(join(cwd, "README.md"), "# test\n");
		this.cwd = cwd;
		this.events = [];
		this.agentStarts = 0;
		this.agentEnds = 0;
		this.proc = spawn(
			PI_BIN,
			["--mode", "rpc", "--model", "anthropic/claude-haiku-4-5", "--session-dir", join(cwd, ".sessions")],
			{ cwd, env: { ...process.env, ADVISOR_NO_REVIEW: "1" } },
		);
		this.proc.stderr.on("data", () => {});
		let buffer = "";
		const decoder = new StringDecoder("utf8");
		this.proc.stdout.on("data", (chunk) => {
			buffer += decoder.write(chunk);
			for (;;) {
				const i = buffer.indexOf("\n");
				if (i === -1) break;
				let line = buffer.slice(0, i);
				buffer = buffer.slice(i + 1);
				if (line.endsWith("\r")) line = line.slice(0, -1);
				if (!line.trim()) continue;
				let ev;
				try {
					ev = JSON.parse(line);
				} catch {
					continue;
				}
				this.events.push(ev);
				if (ev.type === "agent_start") this.agentStarts++;
				if (ev.type === "agent_end") this.agentEnds++;
			}
		});
	}
	send(cmd) {
		this.proc.stdin.write(JSON.stringify(cmd) + "\n");
	}
	prompt(message) {
		this.send({ type: "prompt", message });
	}
	async sleep(ms) {
		return new Promise((r) => setTimeout(r, ms));
	}
	async waitFor(pred, timeoutMs, label) {
		const t0 = Date.now();
		while (Date.now() - t0 < timeoutMs) {
			if (pred()) return true;
			await this.sleep(150);
		}
		throw new Error(`timeout waiting for ${label}`);
	}
	async getMessages() {
		const id = "gm-" + Math.random().toString(36).slice(2);
		const before = this.events.length;
		this.send({ id, type: "get_messages" });
		await this.waitFor(
			() => this.events.slice(before).some((e) => e.type === "response" && e.id === id),
			5000,
			"get_messages response",
		);
		const resp = this.events.slice(before).find((e) => e.type === "response" && e.id === id);
		return resp?.data?.messages || [];
	}
	kill() {
		try {
			this.proc.kill("SIGTERM");
		} catch {}
	}
}

if (process.env.ADVISOR_E2E) {
	test("E2E: a nit is delivered immediately, triggers a turn, and lands in transcript", async () => {
		const pi = new RpcPi();
		try {
			await pi.sleep(2500);
			const before = pi.agentStarts;
			pi.prompt("/advisor test nit NITSENTINEL tidy this later");
			// nits now steer + triggerTurn: an idle agent wakes to act on them.
			await pi.waitFor(() => pi.agentStarts > before, 30000, "nit-triggered agent_start");
			await pi.waitFor(() => pi.agentEnds >= 1, 60000, "triggered turn agent_end");
			const adv = (await pi.getMessages()).find(
				(m) => m.role === "custom" && m.customType === "advisory" && JSON.stringify(m).includes("NITSENTINEL"),
			);
			assert.ok(adv, "nit advisory lands in the transcript as an advisory custom message");
		} finally {
			pi.kill();
		}
	});
} else {
	test("E2E (skipped: set ADVISOR_E2E=1 to run the pi harness)", () => {});
}

// ===========================================================================
// runner
// ===========================================================================

for (const [name, fn] of tests) {
	try {
		await fn();
		passed++;
		console.log(`  ok   ${name}`);
	} catch (err) {
		console.error(`  FAIL ${name}\n       ${err.message}`);
	}
}
console.log(`\n${passed}/${tests.length} passed`);
process.exit(passed === tests.length ? 0 : 1);
