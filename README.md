# llm-pipeline

A Rust library for composing provider-backed LLM payloads with explicit prompt rendering, typed extraction, defensive output parsing, streaming, transport retries, semantic retries, limits, events, and receipts-oriented metadata.

The primary API is payload-oriented:

```text
ExecCtx ──► Payload / LlmCall ──► PayloadOutput<Value> ──► typed parse at the workflow edge
   │              │
   │              ├── OllamaBackend (default)
   │              ├── OpenAiBackend (feature = "openai")
   │              ├── AnthropicBackend (feature = "anthropic")
   │              ├── MockBackend
   │              └── RecordingBackend
   │
   └── Chain composes payloads sequentially; agent-graph supplies branching control flow.
```

> Current crate version: `0.2.1` · Rust 2021 · MIT
>
> This README describes the checked-in implementation. It does not claim provider uptime, model quality, benchmark superiority, or successful external calls without a locally reproduced run.

## Responsibility boundary

| Layer | Responsibility |
| --- | --- |
| `llm-pipeline` | Provider-neutral request/response normalization, payload composition, parsing, retries, streaming, budgets/limits, diagnostics |
| `agent-graph` | Branching graph topology, state, routing, fan-out, interrupts, graph checkpoints |
| `agent-graph-mcp` | MCP transport, graph registry, run lifecycle, durable projections, approvals, witnesses, receipts |
| Provider | Actual model execution, provider authentication, provider-specific quotas and responses |

Use `Chain` when the workflow is sequential. Use `agent-graph` when you need conditional routing, parallel branches, subgraphs, or graph-level pause/resume. Use `agent-graph-mcp` when those capabilities must be exposed through MCP tools.

## Install

The published package uses registry dependencies and is ready for external consumers:

```bash
cargo add llm-pipeline
cargo test -p llm-pipeline
```

To validate the checked-out repository itself:

```bash
cargo check --all-targets
cargo test --all-targets --all-features
```

## Quick start: test without a provider

`MockBackend` makes the first run deterministic and does not contact a live LLM:

```rust
use llm_pipeline::{ExecCtx, LlmCall, MockBackend};
use llm_pipeline::payload::Payload;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
struct MovieReview {
    title: String,
    rating: f64,
    summary: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mock = MockBackend::fixed(
        r#"{"title":"Inception","rating":9.2,"summary":"A mind-bending thriller."}"#,
    );

    let ctx = ExecCtx::builder("http://unused")
        .backend(Arc::new(mock))
        .build();

    let call = LlmCall::new("review", "Review the movie: {input}").expecting_json();
    let output = call.invoke(&ctx, json!("Inception")).await?;
    let review: MovieReview = output.parse_as()?;

    println!("{}: {}/10", review.title, review.rating);
    println!("{}", review.summary);
    Ok(())
}
```

The same path is checked into `examples/mock_example.rs`:

```bash
cargo run -p llm-pipeline --example mock_example
```

## Quick start: Ollama

The default execution backend is `OllamaBackend`. The checked-in `basic_pipeline` example targets an Ollama endpoint and the `llama3.2:3b` model:

```bash
ollama serve
ollama pull llama3.2:3b
cargo run -p llm-pipeline --example basic_pipeline
```

The example expects Ollama at `http://localhost:11434`. Provider startup, model availability, GPU use, latency, and output quality are environment-specific and must be checked locally.

A direct payload call looks like this:

```rust
use llm_pipeline::{ExecCtx, LlmCall};
use llm_pipeline::payload::Payload;
use serde_json::json;

let ctx = ExecCtx::builder("http://localhost:11434")
    .var("audience", "engineers")
    .build();

let call = LlmCall::new(
    "summarize",
    "Summarize this for {audience}: {input}",
)
.with_model("llama3.2:3b")
.expecting_text();

let output = call.invoke(&ctx, json!("some input")).await?;
println!("{}", output.value);
```

## Primary API

### `Payload` and `PayloadOutput`

`Payload` is object-safe and accepts a `serde_json::Value`, allowing heterogeneous workflows:

```rust
pub trait Payload: Send + Sync {
    fn kind(&self) -> &'static str;
    fn name(&self) -> &str;
    fn invoke<'a>(
        &'a self,
        ctx: &'a ExecCtx,
        input: serde_json::Value,
    ) -> BoxFut<'a, Result<PayloadOutput>>;
}
```

`PayloadOutput` keeps both the structured `value` and execution metadata:

- raw response text and optional thinking text;
- model, finish reason, TTFT, cache-hit flag, and normalized token usage;
- parse diagnostics and retry counters;
- response size and wall-clock duration;
- canonical `stack_ids::TraceCtx` on the normal LLM path.

Extract typed data at a workflow boundary rather than making every internal payload share one generic type:

```rust
let output = call.invoke(&ctx, json!("input")).await?;
let typed: MyOutput = output.parse_as()?;
```

### `ExecCtx`

Build one context and share it across a chain or graph. It carries:

- the reusable `reqwest::Client`;
- provider base URL and an `Arc<dyn Backend>`;
- template variables for `{key}` substitution;
- transport backoff configuration;
- optional cancellation flag;
- optional event handler;
- trace context;
- `PipelineLimits`, optional cost model, and optional token budget.

The default backend is Ollama. Use a feature-gated builder method or inject an `Arc<dyn Backend>` when selecting another provider or a test double.

### `LlmCall`

`LlmCall` is the main provider-backed payload. It supports:

- `{input}` plus context-variable prompt templates;
- optional system prompt via `with_system`;
- model selection with `with_model`;
- `LlmConfig` for temperature, max tokens, thinking, JSON mode, and constraints;
- streaming with `with_streaming(true)`;
- a per-call timeout with `with_timeout`;
- output strategies and semantic retry configuration;
- JSON, list, choice, number, bounded-number, text, XML-tag, or custom parsing.

Example with explicit JSON parsing and a per-call timeout:

```rust
use std::time::Duration;
use llm_pipeline::{ExecCtx, LlmCall, LlmConfig};
use llm_pipeline::payload::Payload;
use serde_json::json;

let ctx = ExecCtx::builder("http://localhost:11434").build();
let call = LlmCall::new("classify", "Classify: {input}")
    .with_model("llama3.2:3b")
    .with_config(LlmConfig::default().with_json_mode(true))
    .with_timeout(Duration::from_secs(30))
    .expecting_json();
let output = call.invoke(&ctx, json!("a short request")).await?;
```

### `Chain`

`Chain` executes payloads in order and passes each output `value` to the next payload. It also implements `Payload`, so chains can be nested.

```rust
use llm_pipeline::{Chain, ExecCtx, LlmCall};
use llm_pipeline::payload::Payload;
use serde_json::json;

let chain = Chain::new("draft-and-refine")
    .push(Box::new(LlmCall::new("draft", "Draft: {input}")))
    .push(Box::new(LlmCall::new("refine", "Refine: {input}")));

let ctx = ExecCtx::builder("http://localhost:11434").build();
let final_output = chain.execute(&ctx, json!("topic")).await?;
```

Use `execute_all` when intermediate outputs and their diagnostics are required. An empty chain returns an error.

## Backends and feature flags

| Backend | Availability | Transport shape |
| --- | --- | --- |
| `OllamaBackend` | Default | `/api/generate`, `/api/chat`, NDJSON streaming |
| `OpenAiBackend` | `--features openai` | OpenAI-compatible chat completions and SSE path |
| `AnthropicBackend` | `--features anthropic` | Anthropic Messages API and SSE path |
| `MockBackend` | Always compiled | Fixed or test-controlled responses; no network required |
| `RecordingBackend` | Always compiled | Records backend interactions around another backend |

Feature examples:

```bash
cargo check -p llm-pipeline --features openai
cargo check -p llm-pipeline --features anthropic
cargo test -p llm-pipeline --all-features
```

The `openai` and `anthropic` feature flags compile backend support; they do not create credentials or prove that a provider endpoint is reachable.

## Output parsing and semantic retries

`OutputStrategy` controls how raw model text becomes a `serde_json::Value`:

| Strategy | Result |
| --- | --- |
| `Lossy` (default) | Attempts JSON extraction, then falls back to a string; preserves legacy behavior and usually does not fail on parse shape |
| `Json` | Multi-strategy JSON extraction with repair; parse failure is observable |
| `StringList` | Extracts a JSON/string list |
| `XmlTag(name)` | Extracts a named XML tag body |
| `Choice(values)` | Matches one of the allowed choices |
| `Number` | Extracts a numeric value |
| `NumberInRange(min, max)` | Extracts a number and enforces bounds |
| `Text` | Cleans text output and strips common boilerplate |
| `Custom(fn)` | Caller-defined parser returning `Value` or a parse error |

Prefer a strict strategy when downstream code depends on shape. A successful `Lossy` parse is not the same thing as schema validation.

There are two distinct retry layers:

1. **Transport retry** through `BackoffConfig` for connection failures and configured HTTP statuses such as 429 or 5xx.
2. **Semantic retry** through `RetryConfig` when parsing or output validation fails.

The output records both counters separately. Configure budgets and retry limits so a malformed model response cannot create an unbounded loop.

## Streaming and events

Set `with_streaming(true)` for provider streaming where the selected backend supports it. Attach an `EventHandler` to `ExecCtx` for payload lifecycle and token events:

```rust
use llm_pipeline::events::{Event, FnEventHandler};
use std::sync::Arc;

let ctx = ExecCtx::builder("http://localhost:11434")
    .event_handler(Arc::new(FnEventHandler(|event: Event| {
        if let Event::Token { chunk, .. } = event {
            print!("{chunk}");
        }
    })))
    .build();
```

The final `PayloadOutput` still contains the accumulated normalized response. Streaming does not make a provider call durable or cancellable unless the surrounding context and caller implement those policies.

## Limits, costs, traces, and receipts

`PipelineLimits` bounds pipeline work such as request timeout and maximum tokens per call. `ExecCtx` can also carry:

- an optional global token budget;
- a `CostModel` for cost-update events;
- event handlers for lifecycle instrumentation;
- canonical trace context from `stack-ids`;
- cancellation state checked before payload execution.

These facilities emit runtime metadata and enforce local policies. They do not certify provider billing, pricing, or model behavior. Preserve raw events and provider metadata when a run needs an auditor-rerunnable record.

## Compatibility API

The crate still contains the older `Pipeline`, `Stage`, `PipelineInput`, and helper call API. They remain useful for compatibility and the `basic_pipeline` example, but new code should prefer:

```text
ExecCtx + LlmCall + PayloadOutput
```

The older `call_llm` and `call_llm_chat` helpers are deprecated in source documentation in favor of `LlmCall` with `ExecCtx`, because the newer path supports multiple backends, output strategies, streaming, and retry policy.

## Examples

From the workspace root:

```bash
cargo run -p llm-pipeline --example mock_example
cargo run -p llm-pipeline --example basic_pipeline
cargo run -p llm-pipeline --example payload_chain
cargo run -p llm-pipeline --example streaming_pipeline
cargo run -p llm-pipeline --example context_injection
cargo run -p llm-pipeline --example thinking_mode
cargo run -p llm-pipeline --example anthropic_budget --features anthropic
```

The provider-backed examples need their corresponding local provider/model. `mock_example` is the provider-independent smoke path.

## Verification

```bash
cargo fmt --check -p llm-pipeline
cargo check -p llm-pipeline --all-targets
cargo test -p llm-pipeline
cargo test -p llm-pipeline --all-features
```

A passing unit or integration test proves only the exercised local contract. It does not prove network availability, provider compatibility beyond the test double, output quality, cost accuracy against a live invoice, or production readiness.

## Troubleshooting

### Connection errors

Confirm the endpoint passed to `ExecCtx::builder`, verify the provider is listening, and test the provider independently. The default Ollama endpoint is `http://localhost:11434`.

### Unexpected strings instead of objects

The default output strategy is `Lossy`. Use `expecting_json()` or an explicit `OutputStrategy::Json`, then parse the `PayloadOutput` into the target type and inspect `diagnostics`.

### Requests retry too long

Review both transport `BackoffConfig` and semantic `RetryConfig`, then apply `PipelineLimits` and per-call `with_timeout`. These are separate retry domains.

### A feature-gated backend is missing

Compile with the corresponding feature (`openai` or `anthropic`). The feature enables code; credentials and endpoint configuration remain caller responsibilities.

## License

MIT. See the repository license files for the governing text.
