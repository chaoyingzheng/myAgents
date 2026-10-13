# Product feedback log

Running notes taken while building. Presented in the last minutes of the interview.

## Weave
- **What worked:** the Agents view traces exposed two real bugs within minutes:
  1. The model *described* the computation ("we will now compute...") but made 0 tool calls.
  2. The tool returned the literal text `${M:.2f}` (the model wrote escaped f-string braces), and the model then invented a number.
- **API gap:** `client.get_calls()` returns nothing for Agents-SDK (OTel) traces; you need the separate `get_agent_spans()` / `get_agent_turn()` API. `get_agent_spans()` also omits tool arguments and results, which only `get_agent_turn()` returns. None of this is obvious from the docs.
- The OpenAI Agents SDK's default exporter (to OpenAI) keeps running next to Weave's processor. I had to call `set_trace_processors([])` myself; the integration page doesn't mention this.
- **Project naming:** using the *org* name as the entity fails with "you may not create the generic project ... directly under your org, please try using your team entity". The UI shows the org name prominently, so new users will likely hit this.
- Docs moved from docs.wandb.ai to docs.coreweave.com (301 redirects); search results still point at the old URLs.

## Serverless Inference
- **What worked:** setup is truly 1 key + 1 base_url, and the model list is broad (20 models).
- **gpt-oss-120b:** returns an empty final message when driven by the OpenAI Agents SDK (chat completions + tools). A raw single call with tools works, so it's likely a reasoning/harmony-format issue in multi-step tool loops. A compatibility matrix (tool calling, `tool_choice=required`, parallel tool calls) per model would help.
- **Concurrency limits aren't published.** The usage-limits page says limits apply "per W&B project and per user" and that exceeding them returns HTTP 429, but it gives no numbers and doesn't say whether they differ by plan; the only option is "contact support". Customers can't capacity-plan from this. Ideally the limit would be shown in the billing/usage UI and exposed via response headers (like `x-ratelimit-*`).
- **Model choice mattered a lot:** Llama-3.3-70B wrote buggy code and then invented numbers. Qwen3.6-35B-A3B (3B active params, cheaper) was correct.

## Sandboxes
- **What worked:** it ran out of the box with the W&B key on a free account. Cold start was ~3s for `python:3.11-slim`.
- **Conflicting access docs:** the get-started page says access is "enabled per organization" via support, while the press release says "use your existing W&B API key and start in minutes".
- **Two install paths:** `wandb[sandbox]` / `wandb.sandbox` in older docs vs `cwsandbox[wandb]` + `AuthStrategy.WANDB` in current docs, and both show up in search. Confusing on day one.
- **Missing reference:** the Python client docs page lacks an exec timeout and error-class reference. I read the SDK source to find `exec(timeout_seconds=...)` and `SandboxTimeoutError`.

## Agent Lens / ARIA / W&B MCP server
-
