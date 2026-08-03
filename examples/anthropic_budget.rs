//! Anthropic + cost-model + token-budget example.
//!
//! Run with:
//!
//! ```bash
//! ANTHROPIC_API_KEY=sk-xxx cargo run -p llm-pipeline --example anthropic_budget --features anthropic
//! ```

use llm_pipeline::payload::Payload;
use llm_pipeline::{backend::AnthropicBackend, CostModel, ExecCtx, LlmCall};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("set ANTHROPIC_API_KEY");

    let ctx = ExecCtx::builder("https://api.anthropic.com")
        .backend(std::sync::Arc::new(AnthropicBackend::new(api_key)))
        .with_token_budget(4096)
        .with_cost_model(CostModel::new(3.0, 15.0, 0.0))
        .build();

    let call = LlmCall::new(
        "haiku",
        "Write a haiku about {topic}. Reply in JSON as {{\"haiku\": string}}.",
    )
    .with_model("claude-3-5-sonnet-20241022")
    .expecting_json();

    let output = call.invoke(&ctx, json!({"topic": "Rust"})).await?;
    println!("{}", serde_json::to_string_pretty(&output.value)?);

    Ok(())
}
