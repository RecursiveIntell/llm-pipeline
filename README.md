# llm-pipeline

Production-grade LLM call execution, output parsing, and retry for Rust.

[![Crates.io](https://img.shields.io/crates/v/llm-pipeline.svg)](https://crates.io/crates/llm-pipeline)
[![docs.rs](https://docs.rs/llm-pipeline/badge.svg)](https://docs.rs/llm-pipeline)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What this crate does

- **LLM call execution** via pluggable backends (Ollama, OpenAI-compatible, mock)
- **Defensive output parsing** — extracts JSON from markdown fences, repairs trailing commas / single quotes, handles `<think>` blocks
- **Semantic retry** — re-prompts the LLM with error context on parse failure
- **Transport retry** with exponential backoff, jitter, and `Retry-After` support
- **Sequential chaining** — pipe payload outputs through multi-step workflows
- **Streaming** with per-token event callbacks
- **Typed extraction** — `output.parse_as::<T>()` at workflow edges

## What this crate does NOT do

- Graph orchestration (routing, branching, loops, checkpoints) — use a graph runtime
- Prompt engineering or prompt management
- Embeddings, vector stores, or RAG

## Quick Start (no LLM required)

```rust
use llm_pipeline::{ExecCtx, LlmCall, MockBackend};
use llm_pipeline::payload::Payload;
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

#[derive(Debug, Deserialize)]
struct Review { title: String, rating: f64 }

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = ExecCtx::builder("http://localhost:11434").build();

    let chain = Chain::new("analyze")
        .push(Box::new(
            LlmCall::new("draft", "Analyze: {input}")
                .with_config(LlmConfig::default().with_json_mode(true))
        ))
        .push(Box::new(
            LlmCall::new("refine", "Refine this analysis: {input}")
                .with_config(LlmConfig::default().with_json_mode(true))
        ));

    let output = chain.execute(&ctx, json!("Rust's ownership model")).await?;
    println!("{}", output.value);
    Ok(())
}
```

## Feature flags

| Feature  | Default | Description |
|----------|---------|-------------|
| `yaml`   | off     | YAML output parsing via `serde_yaml` |
| `openai` | off     | OpenAI-compatible backend (`OpenAiBackend`, SSE decoder) |

```toml
[dependencies]
llm-pipeline = { version = "0.1", features = ["openai"] }
```

## Output strategies

| Strategy | Use case |
|----------|----------|
| `Lossy` (default) | Best-effort JSON extraction from messy output |
| `Json` | Full multi-strategy extraction with repair |
| `StringList` | Extract `["item1", "item2"]` arrays |
| `Choice(vec)` | Match output to one of N options |
| `Number` / `NumberInRange` | Extract numeric values |
| `Text` | Clean text (strip preamble) |
| `XmlTag(tag)` | Extract `<tag>content</tag>` |
| `Custom(fn)` | Your own parser |

## Backends

- **`OllamaBackend`** (default) — Ollama native API (`/api/generate`, `/api/chat`)
- **`OpenAiBackend`** (feature `openai`) — Any OpenAI-compatible API (`/v1/chat/completions`)
- **`MockBackend`** — Canned responses for testing

## Examples

```bash
cargo run --example mock_example        # No LLM needed
cargo run --example payload_chain       # Requires Ollama
cargo run --example basic_pipeline      # Requires Ollama
cargo run --example streaming_pipeline  # Requires Ollama
```

## License

MIT
