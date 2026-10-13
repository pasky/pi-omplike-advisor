import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function deferred() {
	let resolve;
	const promise = new Promise((r) => { resolve = r; });
	return { promise, resolve };
}

// Real Agent lifecycle, deterministic in-memory streams. No provider or network.
export function registerAdvisorResetTests(test, { AdvisorRuntime, AdviseTool, CoreAgent, EventStream, default: installAdvisor }) {
	function fixture() {
		const started = deferred();
		const release = deferred();
		const contexts = [];
		const message = (text, stopReason = "stop") => ({
			role: "assistant", content: [{ type: "text", text }],
			api: "openai-responses", provider: "mock", model: "mock",
			usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
			stopReason, timestamp: Date.now(),
		});
		const agent = new CoreAgent({
			streamFn: (_model, context) => {
				contexts.push(JSON.stringify(context.messages));
				const stream = new EventStream(
					(event) => event.type === "done" || event.type === "error",
					(event) => event.type === "done" ? event.message : event.error,
				);
				if (contexts.length === 1) {
					started.resolve();
					// Deliberately hold completion after abort. abort() is not waitForIdle().
					void release.promise.then(() => stream.push({ type: "error", reason: "aborted", error: message("OLD_RESPONSE", "aborted") }));
				} else {
					queueMicrotask(() => stream.push({ type: "done", reason: "stop", message: message("NEW_RESPONSE") }));
				}
				return stream;
			},
		});
		const debug = [];
		const runtime = new AdvisorRuntime(agent, new AdviseTool(() => false), 0, (...args) => debug.push(args.join(" ")));
		async function cleanup() {
			runtime.dispose();
			release.resolve();
			await agent.waitForIdle();
		}
		return { agent, runtime, contexts, debug, started, release, cleanup };
	}

	test("extension: an enabled advisor replaces a runtime disposed by a genuine reset failure", async () => {
		const root = mkdtempSync(join(tmpdir(), "pi-advisor-reset-rebuild-"));
		const previousDir = process.env.PI_CODING_AGENT_DIR;
		const originalReset = CoreAgent.prototype.reset;
		const handlers = new Map();
		const commands = new Map();
		let lookups = 0;
		const model = { provider: "mock", id: "mock", api: "openai-responses", reasoning: false, contextWindow: 1000 };
		const ctx = {
			cwd: root, model, hasUI: false,
			modelRegistry: { find: () => { lookups++; return model; } },
			sessionManager: { getSessionId: () => "reset-test", getBranch: () => [], getEntries: () => [] },
			ui: { notify() {}, setStatus() {} },
		};
		try {
			process.env.PI_CODING_AGENT_DIR = root;
			installAdvisor({
				on: (name, fn) => handlers.set(name, fn),
				registerCommand: (name, value) => commands.set(name, value),
				registerMessageRenderer() {}, appendEntry() {},
				events: { on() {} },
			});
			await handlers.get("session_start")({ reason: "startup" }, ctx);
			await commands.get("advisor").handler("on", ctx);
			assert.ok(lookups > 0, "first runtime was built");
			const before = lookups;
			CoreAgent.prototype.reset = () => { throw new Error("synthetic reset failure"); };
			await handlers.get("session_compact")({}, ctx);
			CoreAgent.prototype.reset = originalReset;
			await commands.get("advisor").handler("status", ctx);
			assert.ok(lookups > before, "disposed runtime must not be returned from the cache");
		} finally {
			CoreAgent.prototype.reset = originalReset;
			await handlers.get("session_shutdown")?.({}, ctx);
			if (previousDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
			else process.env.PI_CODING_AGENT_DIR = previousDir;
			rmSync(root, { recursive: true, force: true });
		}
	});

	test("runtime: compaction during a real Agent review resets only after it settles", async () => {
		const f = fixture();
		try {
			f.runtime.push("OLD_TRANSCRIPT");
			await f.started.promise;
			const waiter = f.runtime.waitUntilSettled(2000);
			f.runtime.enqueueAdvice("OLD_ADVICE", "blocker");
			f.runtime.reset();
			assert.equal(await waiter, "aborted", "old boundary waiters must cancel immediately");
			assert.equal(f.runtime.acceptingAdvice, false, "old callbacks must be fenced immediately");
			assert.equal(f.runtime.hasHighPriority, false);
			assert.equal(f.runtime.disposed, false, "a busy agent is not a fatal reset failure");
			f.runtime.push("FRESH_TRANSCRIPT");
			assert.equal(f.contexts.length, 1, "new review must wait for the old stream");
			f.release.resolve();
			assert.equal(await f.runtime.waitUntilSettled(2000), "settled");
			assert.equal(f.contexts.length, 2);
			assert.match(f.contexts[1], /FRESH_TRANSCRIPT/);
			assert.doesNotMatch(f.contexts[1], /OLD_TRANSCRIPT|OLD_RESPONSE|OLD_ADVICE/);
			assert.equal(f.runtime.usage.input, 1, "discarded history must not contaminate reset accounting");
			assert.deepEqual(f.debug.filter((line) => /reset failed/.test(line)), []);
		} finally { await f.cleanup(); }
	});

	test("runtime: a deferred reset clears old history even without another turn", async () => {
		const f = fixture();
		try {
			f.runtime.push("OLD_TRANSCRIPT");
			await f.started.promise;
			f.runtime.reset();
			f.release.resolve();
			assert.equal(await f.runtime.waitUntilSettled(2000), "settled");
			assert.equal(f.runtime.disposed, false);
			assert.deepEqual(f.agent.state.messages, []);
			assert.equal(f.runtime.usage.input, 0);
			assert.equal(f.contexts.length, 1);
		} finally { await f.cleanup(); }
	});

	test("runtime: repeated resets keep only work queued after the latest reset", async () => {
		const f = fixture();
		try {
			f.runtime.push("OLD_TRANSCRIPT");
			await f.started.promise;
			f.runtime.reset();
			f.runtime.push("DISCARDED_DELTA");
			f.runtime.reset();
			f.runtime.push("LATEST_DELTA");
			f.release.resolve();
			assert.equal(await f.runtime.waitUntilSettled(2000), "settled");
			assert.equal(f.contexts.length, 2);
			assert.match(f.contexts[1], /LATEST_DELTA/);
			assert.doesNotMatch(f.contexts[1], /OLD_TRANSCRIPT|DISCARDED_DELTA/);
		} finally { await f.cleanup(); }
	});

	test("runtime: reset during a retry delay drops the failed batch before fresh work", async () => {
		const failed = deferred();
		const contexts = [];
		let resets = 0;
		const agent = {
			state: { messages: [], model: {} },
			async prompt(messages) {
				contexts.push(JSON.stringify([...this.state.messages, ...messages]));
				this.state.messages.push(...messages);
				if (contexts.length === 1) throw new Error("synthetic provider failure");
				this.state.messages.push({ role: "assistant", content: [], stopReason: "stop" });
			},
			abort() {},
			reset() { resets++; this.state.messages = []; },
		};
		const runtime = new AdvisorRuntime(agent, new AdviseTool(() => false), 25, (event) => {
			if (event === "advisor prompt threw") failed.resolve();
		});
		try {
			runtime.push("FAILED_OLD_BATCH");
			await failed.promise; // drain has requeued the failed batch and entered its retry timer
			runtime.reset();
			runtime.push("FRESH_BATCH");
			assert.equal(await runtime.waitUntilSettled(2000), "settled");
			assert.equal(resets, 1);
			assert.equal(contexts.length, 2);
			assert.match(contexts[1], /FRESH_BATCH/);
			assert.doesNotMatch(contexts[1], /FAILED_OLD_BATCH/);
		} finally { runtime.dispose(); }
	});

	test("runtime: disposal while reset is pending never restarts a review", async () => {
		const f = fixture();
		try {
			f.runtime.push("OLD_TRANSCRIPT");
			await f.started.promise;
			f.runtime.reset();
			f.runtime.push("NEVER_REVIEW");
			f.runtime.dispose();
			f.runtime.reset();
			f.release.resolve();
			await f.agent.waitForIdle();
			assert.equal(await f.runtime.waitUntilSettled(2000), "aborted");
			assert.equal(f.contexts.length, 1);
			assert.equal(f.runtime.disposed, true);
		} finally { await f.cleanup(); }
	});
}
