# llm-pipeline

Production-grade LLM call execution, defensive output parsing, and retry for Rust.

[![Crates.io](https://img.shields.io/crates/v/llm-pipeline.svg)](https://crates.io/crates/llm-pipeline)
[![docs.rs](https://docs.rs/llm-pipeline/badge.svg)](https://docs.rs/llm-pipeline)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What this crate does

Handles everything that happens *inside* an LLM workflow node: render a prompt, call a provider, parse the response, retry if it's broken. Your orchestrator (graph runtime, agent loop, or a simple `for` loop) decides *which* node runs next — this crate makes sure each node produces usable output.

- **Pluggable backends** — Ollama and OpenAI-compatible APIs, plus a `MockBackend` for testing without infrastructure
- **Defensive output parsing** — multi-strategy extraction pipeline that handles markdown fences, `<think>` blocks, embedded JSON in prose, and deterministic repair of malformed output (trailing commas, single quotes, Python literals, unquoted keys, unclosed brackets)
- **Two-tier retry** — transport-level backoff (429/5xx with jitter and `Retry-After` support) and semantic retry that re-prompts the LLM with error context and temperature cool-down when parsing can't salvage the output
- **Sequential chaining** — compose payloads into multi-step workflows with `Chain`
- **Streaming** — per-token callbacks with a buffered NDJSON decoder that handles chunk-boundary splits
- **Typed extraction** — `output.parse_as::<T>()` at workflow edges

## What this crate does NOT do

- Graph orchestration (routing, branching, parallel execution, checkpoints)
- Prompt engineering or prompt management
- Embeddings, vector stores, or RAG

## Quick start — no LLM required

```rust
use llm_pipeline::{ExecCtx, LlmCall, MockBackend};
use llm_pipeline::payload::Payload;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
struct Review { title: String, rating: f64 }

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mock = MockBackend::fixed(r#"{"title": "Inception", "rating": 9.2}"#);
    let ctx = ExecCtx::builder("http://unused")
        .backend(Arc::new(mock))
        .build();

    let call = LlmCall::new("review", "Review: {input}").expecting_json();
    let output = call.invoke(&ctx, json!("Inception")).await?;
    let review: Review = output.parse_as()?;

    println!("{}: {}/10", review.title, review.rating);
    Ok(())
}
```

## With a live Ollama instance

```rust,no_run
use llm_pipeline::{ExecCtx, LlmCall, Chain, LlmConfig};
use llm_pipeline::payload::Payload;
use llm_pipeline::retry::RetryConfig;
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ctx = ExecCtx::builder("http://localhost:11434").build();

    let call = LlmCall::new("analyze", "Analyze this topic and return JSON: {input}")
        .with_model("llama3.2:3b")
        .with_system("You are a research analyst. Always respond in JSON.")
        .with_config(LlmConfig::default().with_json_mode(true))
        .expecting_json()
        .with_retry(RetryConfig::new(2).requiring_keys(&["summary", "confidence"]));

    let output = call.invoke(&ctx, json!("Rust's ownership model")).await?;
    println!("{}", output.value);
    Ok(())
}
```

## Core concepts

**`ExecCtx`** — shared execution context built once and passed to every payload. Carries the HTTP client, backend, base URL, template variables, cancellation flag, and optional event handler.

**`LlmCall`** — the primary payload. Renders a prompt template, calls the backend, parses the response through the configured output strategy, and optionally retries. Builder methods configure everything: `.with_model()`, `.with_system()`, `.with_streaming(true)`, `.expecting_json()`, `.with_retry()`.

**`Payload`** — object-safe trait (`Box<dyn Payload>`) that takes a `serde_json::Value` input and returns a `PayloadOutput`. `LlmCall` and `Chain` both implement it, so chains can nest.

**`PayloadOutput`** — wraps the parsed `Value`, raw response text, optional thinking content, and `ParseDiagnostics` recording which strategy succeeded, retry counts, and backoff time.

**`Chain`** — sequential composition. Pipes each payload's output value as the next payload's input. Respects cancellation between steps.

## How output parsing works

When an LLM responds, the text goes through a parsing pipeline before the retry system ever sees it:

1. **Preprocess** — strip `<think>...</think>` blocks, trim whitespace
2. **Extract** — try direct parse, then ```` ```json ```` code blocks, then bracket-matching `{...}` or `[...]` from prose
3. **Repair** — if extraction found a candidate but it's malformed, apply deterministic fixes: strip comments, replace Python literals (`True`/`False`/`None`), remove trailing commas, swap single quotes for double, quote bare keys, close unclosed brackets, escape raw newlines
4. **Auto-complete** — for truncated streaming output, close unclosed strings and brackets

If all of that fails and `RetryConfig` is set, *then* the semantic retry kicks in: the original prompt, the bad response, and the parse error are sent back to the LLM as a correction conversation, with temperature reduced by 0.2 per attempt.

## Output strategies

Configure with builder methods on `LlmCall`:

| Method | Strategy | Returns |
|--------|----------|---------|
| *(default)* | `Lossy` | Best-effort JSON extraction, falls back to `Value::String` — never fails |
| `.expecting_json()` | `Json` | Full extraction + repair pipeline, can fail → triggers retry |
| `.expecting_list()` | `StringList` | `["item1", "item2"]` arrays |
| `.expecting_choice(vec![...])` | `Choice` | Matched option from valid set (case-insensitive, handles prose/bold/quotes) |
| `.expecting_number()` | `Number` | Numeric extraction from "Score: 8.5", "8/10", prose |
| `.expecting_number_in_range(1.0, 10.0)` | `NumberInRange` | Bounded numeric extraction |
| `.expecting_text()` | `Text` | Clean prose with boilerplate stripping ("Sure!", "Here's...") |
| `.with_output_strategy(XmlTag("tag".into()))` | `XmlTag` | Content from `<tag>...</tag>` |
| `.with_output_strategy(Custom(arc_fn))` | `Custom` | Your own `fn(&str) -> Result<Value, ParseError>` |

## Transport retry

Configure backoff on `ExecCtx` for automatic retry of transient HTTP errors:

```rust,no_run
use llm_pipeline::{ExecCtx, BackoffConfig};

// Presets
let ctx = ExecCtx::builder("http://localhost:11434")
    .backoff(BackoffConfig::standard())    // 3 retries, 1s initial, 2x multiplier
    .build();

let ctx = ExecCtx::builder("https://api.openai.com")
    .backoff(BackoffConfig::interactive()) // 2 retries, 500ms initial, 10s max
    .build();

let ctx = ExecCtx::builder("https://api.openai.com")
    .backoff(BackoffConfig::aggressive())  // 5 retries, 500ms initial, 120s max
    .build();
```

Retries 429, 500, 502, 503, 504 with full jitter by default. Respects `Retry-After` headers. Emits `Event::TransportRetry` for observability.

## Streaming

```rust,no_run
use llm_pipeline::{ExecCtx, LlmCall};
use llm_pipeline::payload::Payload;
use llm_pipeline::events::{Event, FnEventHandler};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ctx = ExecCtx::builder("http://localhost:11434")
        .event_handler(Arc::new(FnEventHandler(|event: Event| {
            if let Event::Token { chunk, .. } = event {
                print!("{}", chunk);
            }
        })))
        .build();

    let call = LlmCall::new("stream", "Tell me about {input}")
        .with_streaming(true);

    let output = call.invoke(&ctx, json!("black holes")).await?;
    println!("\n---\nFull response: {}", output.raw_response);
    Ok(())
}
```

The streaming path uses `StreamingDecoder`, a buffered NDJSON framer that handles the common case where a JSON line is split across TCP chunks. On stream end, it attempts auto-completion of truncated JSON.

## Template variables

Prompt templates use `{key}` placeholders. `{input}` is always the payload input. Additional variables come from `ExecCtx`:

```rust,no_run
use llm_pipeline::{ExecCtx, LlmCall};

let ctx = ExecCtx::builder("http://localhost:11434")
    .var("domain", "astrophysics")
    .var("audience", "graduate students")
    .build();

// Both {domain} and {audience} are substituted, {input} comes from invoke()
let call = LlmCall::new("explain", "As a {domain} expert, explain {input} for {audience}.");
```

Use `{{` and `}}` to include literal braces (useful for JSON schemas in prompts).

## Cancellation

```rust,no_run
use llm_pipeline::ExecCtx;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

let cancel = Arc::new(AtomicBool::new(false));
let ctx = ExecCtx::builder("http://localhost:11434")
    .cancellation(Some(cancel.clone()))
    .build();

// From another task:
// cancel.store(true, std::sync::atomic::Ordering::Relaxed);
```

Checked before each payload invocation, between chain steps, and between retry attempts.

## Backends

| Backend | Protocol | Feature |
|---------|----------|---------|
| `OllamaBackend` | `/api/generate`, `/api/chat` (NDJSON streaming) | *(default)* |
| `OpenAiBackend` | `/v1/chat/completions` (SSE streaming) | `openai` |
| `MockBackend` | Canned responses, cycles when exhausted | *(always available)* |

Base URLs are normalized at build time — passing `http://localhost:11434/api` or `https://api.openai.com/v1` won't double the path segments.

## Feature flags

| Feature  | Default | Adds |
|----------|---------|------|
| `openai` | off     | `OpenAiBackend`, SSE decoder |
| `yaml`   | off     | YAML output parsing via `serde_yaml` |

```toml
[dependencies]
llm-pipeline = { version = "0.1", features = ["openai"] }
```

## Diagnostics

Every `PayloadOutput` includes `ParseDiagnostics`:

```rust,ignore
if let Some(diag) = &output.diagnostics {
    println!("Strategy: {:?}", diag.strategy);       // e.g. Some("json")
    println!("Parse OK: {}", diag.ok());              // true if no parse_error
    println!("Repaired: {}", diag.repaired);           // trailing comma fix, etc.
    println!("Retries: {}", diag.retry_attempts);      // semantic retry count
    println!("Transport retries: {}", diag.transport_retries);
    println!("Backoff time: {}ms", diag.backoff_total_ms);
}
```

## Examples

```bash
cargo run --example mock_example        # No LLM needed
cargo run --example basic_pipeline      # Requires Ollama
cargo run --example streaming_pipeline  # Requires Ollama
cargo run --example payload_chain       # Requires Ollama
cargo run --example context_injection   # Requires Ollama
cargo run --example thinking_mode       # Requires Ollama + thinking model
```

## License

MIT
