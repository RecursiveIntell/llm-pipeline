//! Backend for the Anthropic Messages API.
//!
//! Endpoint: `/v1/messages` (always chat mode).
//! Streaming: SSE with `data: {"type": "content_block_delta", "delta": {"text": "token"}}`.

use super::sse::SseDecoder;
use super::{Backend, LlmRequest, LlmResponse, ProviderMeta, Role};
use crate::constraints::GenerationConstraint;
use crate::error::{PipelineError, Result};
use crate::payload::TokenUsage;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};

/// Backend for the Anthropic Messages API.
///
/// # Example
///
/// ```
/// use llm_pipeline::backend::AnthropicBackend;
///
/// let backend = AnthropicBackend::new("sk-ant-...");
/// ```
#[derive(Clone)]
pub struct AnthropicBackend {
    /// Anthropic API key.
    pub(crate) api_key: String,
    /// Optional API version. Defaults to `2023-06-01`.
    pub(crate) api_version: String,
}

impl std::fmt::Debug for AnthropicBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicBackend")
            .field(
                "api_key",
                &if self.api_key.len() > 6 {
                    format!("{}***", &self.api_key[..6])
                } else {
                    "***".to_string()
                },
            )
            .field("api_version", &self.api_version)
            .finish()
    }
}

impl AnthropicBackend {
    /// Create a new Anthropic backend with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            api_version: "2023-06-01".to_string(),
        }
    }

    /// Set a custom Anthropic API version header.
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Build the messages array for the Anthropic request.
    fn build_messages(request: &LlmRequest) -> Vec<Value> {
        let mut messages = Vec::new();

        // Anthropic does not use a top-level system prompt; it uses a system field.
        // Prior conversation history is preserved in messages.
        for msg in &request.messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "user", // Anthropic disallows system role in messages
            };
            messages.push(json!({"role": role, "content": msg.content}));
        }

        // Add the current user prompt
        messages.push(json!({"role": "user", "content": request.prompt}));

        messages
    }

    /// Build the request body for `/v1/messages`.
    fn build_body(request: &LlmRequest, stream: bool, max_tokens_limit: Option<u32>) -> Value {
        let max_tokens = max_tokens_limit
            .map(|limit| request.config.max_tokens.min(limit))
            .unwrap_or(request.config.max_tokens);
        let mut body = json!({
            "model": request.model,
            "messages": Self::build_messages(request),
            "max_tokens": max_tokens,
            "stream": stream,
        });

        // Anthropic uses `temperature` only when non-null.
        if request.config.temperature > 0.0 {
            body["temperature"] = json!(request.config.temperature);
        }

        // Structured output constraints: Anthropic supports JSON schema via tools
        // in a limited way, but for simplicity we encode JSON schema as a system
        // instruction and rely on the caller. Native tool use is out of scope here.
        match &request.constraint {
            GenerationConstraint::None => {}
            GenerationConstraint::JsonSchema(_) => {
                // Anthropic does not have a direct response_format equivalent;
                // caller must use prompt engineering or tool use.
            }
            GenerationConstraint::Grammar(_) | GenerationConstraint::Regex(_) => {
                // Not supported by Anthropic Messages API.
            }
        }

        body
    }

    /// Send a request to the Anthropic Messages API.
    async fn send_request(
        &self,
        client: &Client,
        base_url: &str,
        request: &LlmRequest,
        stream: bool,
    ) -> Result<(Value, u16, ProviderMeta)> {
        let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));
        let body = Self::build_body(request, stream, None);
        let timeout = request
            .request_timeout
            .unwrap_or_else(|| std::time::Duration::from_secs(300));

        let mut builder = client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header("content-type", "application/json")
            .timeout(timeout);

        if stream {
            builder = builder.header("accept", "text/event-stream");
        }

        let response = builder
            .json(&body)
            .send()
            .await
            .map_err(PipelineError::Request)?;

        let status = response.status().as_u16();
        let provider_meta = ProviderMeta::from_headers_and_json(response.headers(), None);

        let text = response.text().await.map_err(PipelineError::Request)?;

        let json_resp: Value = serde_json::from_str(&text).map_err(|e| {
            PipelineError::InvalidConfig(format!(
                "failed to parse anthropic response (status {status}): {e}; body: {text}"
            ))
        })?;

        Ok((json_resp, status, provider_meta))
    }

    /// Extract token usage from an Anthropic response.
    fn extract_token_usage(json: &Value) -> Option<TokenUsage> {
        let usage = json.get("usage")?;
        Some(TokenUsage::new(
            usage
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
            usage
                .get("output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32,
        ))
    }

    /// Extract the finish reason from an Anthropic response.
    fn extract_finish_reason(json: &Value) -> Option<String> {
        json.get("stop_reason")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Extract generated text from an Anthropic response.
    fn extract_text(json: &Value) -> String {
        json.get("content")
            .and_then(|c| c.as_array())
            .map(|blocks| {
                blocks
                    .iter()
                    .filter_map(|b| {
                        if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                            b.get("text").and_then(|t| t.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default()
    }
}

#[async_trait]
impl Backend for AnthropicBackend {
    async fn complete(
        &self,
        client: &Client,
        base_url: &str,
        request: &LlmRequest,
    ) -> Result<LlmResponse> {
        let (json_resp, status, provider_meta) =
            self.send_request(client, base_url, request, false).await?;

        let text = Self::extract_text(&json_resp);
        let mut provider_meta = provider_meta;
        provider_meta.raw = Some(json_resp.clone());

        Ok(LlmResponse {
            text,
            status,
            metadata: Some(json_resp.clone()),
            provider_meta,
            token_usage: Self::extract_token_usage(&json_resp),
            finish_reason: Self::extract_finish_reason(&json_resp),
            ttft_ms: None,
            cache_hit: false,
        })
    }

    async fn complete_streaming(
        &self,
        client: &Client,
        base_url: &str,
        request: &LlmRequest,
        on_token: &mut (dyn FnMut(String) + Send),
    ) -> Result<LlmResponse> {
        let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));
        let body = Self::build_body(request, true, None);
        let timeout = request
            .request_timeout
            .unwrap_or_else(|| std::time::Duration::from_secs(300));

        let builder = client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", &self.api_version)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .timeout(timeout);

        let response = builder
            .json(&body)
            .send()
            .await
            .map_err(PipelineError::Request)?;

        let status = response.status().as_u16();
        let provider_meta = ProviderMeta::from_headers_and_json(response.headers(), None);

        let mut accumulated = String::new();
        let mut stream = response.bytes_stream();
        let mut decoder = SseDecoder::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(PipelineError::Request)?;
            let values = decoder.decode(&chunk);
            for parsed in values {
                if parsed.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                    if let Some(text) = parsed
                        .get("delta")
                        .and_then(|d| d.get("text"))
                        .and_then(|t| t.as_str())
                    {
                        accumulated.push_str(text);
                        on_token(text.to_string());
                    }
                }
            }
        }

        // Flush any trailing SSE data.
        for parsed in decoder.flush() {
            if parsed.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                if let Some(text) = parsed
                    .get("delta")
                    .and_then(|d| d.get("text"))
                    .and_then(|t| t.as_str())
                {
                    accumulated.push_str(text);
                    on_token(text.to_string());
                }
            }
        }

        Ok(LlmResponse {
            text: accumulated,
            status,
            metadata: None,
            provider_meta,
            token_usage: None,
            finish_reason: None,
            ttft_ms: None,
            cache_hit: false,
        })
    }

    fn name(&self) -> &'static str {
        "anthropic"
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_json_schema(&self) -> bool {
        false
    }

    fn supports_grammar(&self) -> bool {
        false
    }

    fn supports_regex(&self) -> bool {
        false
    }

    fn max_context_tokens(&self) -> Option<u32> {
        Some(200_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ChatMessage;
    use crate::backend::LlmConfig;

    #[test]
    fn test_anthropic_build_messages_preserves_history() {
        let request = LlmRequest {
            model: "claude-3-haiku".into(),
            system_prompt: Some("You are helpful".into()),
            prompt: "hello".into(),
            messages: vec![
                ChatMessage {
                    role: Role::User,
                    content: "prior".into(),
                },
                ChatMessage {
                    role: Role::Assistant,
                    content: "answer".into(),
                },
            ],
            config: LlmConfig::default(),
            max_tokens_limit: None,
            stream: false,
            request_timeout: None,
            constraint: GenerationConstraint::None,
        };

        let messages = AnthropicBackend::build_messages(&request);
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"], "hello");
    }

    #[test]
    fn test_anthropic_extract_text_and_usage() {
        let json = json!({
            "content": [
                {"type": "text", "text": "Hello, world!"}
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            },
            "stop_reason": "end_turn"
        });

        assert_eq!(AnthropicBackend::extract_text(&json), "Hello, world!");
        let usage = AnthropicBackend::extract_token_usage(&json).unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(
            AnthropicBackend::extract_finish_reason(&json),
            Some("end_turn".to_string())
        );
    }
}
