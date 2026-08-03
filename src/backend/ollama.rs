//! Backend for Ollama's native API.
//!
//! [`OllamaBackend`] translates normalized [`LlmRequest`]s
//! into Ollama's `/api/generate` and `/api/chat` endpoints.
//! Streaming uses NDJSON with `{"response": "token"}` per line.
//!
//! This is the default backend and preserves all existing behavior.

use super::{Backend, LlmRequest, LlmResponse, ProviderMeta, Role};
use crate::constraints::GenerationConstraint;
use crate::error::Result;
use crate::payload::TokenUsage;
use crate::streaming::StreamingDecoder;
use crate::PipelineError;
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};

/// Backend for Ollama's native API.
///
/// Endpoints: `/api/generate` (prompt-only), `/api/chat` (with system prompt or messages).
/// Streaming: NDJSON with `{"response": "token"}` per line.
///
/// This is the default backend. Existing code using `ExecCtx::builder("url").build()`
/// uses this backend automatically.
///
/// # Endpoint selection
///
/// Uses `/api/chat` when ANY of:
/// - `system_prompt` is set (non-empty)
/// - `messages` are present (retry with history)
///
/// Uses `/api/generate` when:
/// - No system prompt AND no message history (prompt-only mode)
#[derive(Debug, Clone)]
pub struct OllamaBackend;

impl OllamaBackend {
    /// Build the Ollama `options` object from the LlmConfig.
    fn build_options(request: &LlmRequest, max_tokens_limit: Option<u32>) -> Value {
        let num_predict = max_tokens_limit
            .map(|limit| request.config.max_tokens.min(limit))
            .unwrap_or(request.config.max_tokens);
        let mut opts = json!({
            "temperature": request.config.temperature,
            "num_predict": num_predict,
        });
        if request.config.thinking {
            opts["extended_thinking"] = json!(true);
        }
        if let Some(ref custom) = request.config.options {
            if let (Some(base), Some(extra)) = (opts.as_object_mut(), custom.as_object()) {
                for (k, v) in extra {
                    base.insert(k.clone(), v.clone());
                }
            }
        }
        opts
    }

    /// Apply a structured generation constraint to an Ollama body.
    fn apply_constraint(body: &mut Value, constraint: &GenerationConstraint) {
        match constraint {
            GenerationConstraint::None => {}
            GenerationConstraint::JsonSchema(schema) => {
                body["format"] = schema.clone();
            }
            GenerationConstraint::Grammar(grammar) => {
                body["format"] = json!("grammar");
                body["grammar"] = json!(grammar);
            }
            GenerationConstraint::Regex(regex) => {
                body["format"] = json!("regex");
                body["regex"] = json!(regex);
            }
        }
    }

    /// Whether this request should use `/api/chat` (vs `/api/generate`).
    fn use_chat(request: &LlmRequest) -> bool {
        request
            .system_prompt
            .as_ref()
            .is_some_and(|s| !s.is_empty())
            || !request.messages.is_empty()
    }

    /// Build the JSON body for `/api/generate`.
    fn build_generate_body(request: &LlmRequest, stream: bool) -> Value {
        let mut body = json!({
            "model": request.model,
            "prompt": request.prompt,
            "stream": stream,
            "options": Self::build_options(request, None),
        });
        if request.config.json_mode {
            body["format"] = json!("json");
        }
        Self::apply_constraint(&mut body, &request.constraint);
        body
    }

    /// Build the JSON body for `/api/chat`.
    fn build_chat_body(request: &LlmRequest, stream: bool) -> Value {
        let mut messages = Vec::new();

        // System prompt
        if let Some(ref sys) = request.system_prompt {
            if !sys.is_empty() {
                messages.push(json!({"role": "system", "content": sys}));
            }
        }

        // Prior conversation history (for retry)
        for msg in &request.messages {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            messages.push(json!({"role": role, "content": msg.content}));
        }

        // Current user prompt (only if no messages — if messages are present,
        // the prompt is already the last user message in the history)
        if request.messages.is_empty() {
            messages.push(json!({"role": "user", "content": request.prompt}));
        }

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "stream": stream,
            "options": Self::build_options(request, None),
        });
        if request.config.json_mode {
            body["format"] = json!("json");
        }
        Self::apply_constraint(&mut body, &request.constraint);
        body
    }

    /// Parse a Retry-After header value as seconds.
    fn parse_retry_after(value: &str) -> Option<std::time::Duration> {
        // Try integer seconds first
        if let Ok(secs) = value.trim().parse::<u64>() {
            return Some(std::time::Duration::from_secs(secs));
        }
        None
    }

    /// Send a non-streaming request and parse the response.
    ///
    /// If `request_timeout` is `Some`, it is applied at the `RequestBuilder`
    /// level so this individual request can have a different timeout from the
    /// client-wide default.
    async fn send_request(
        client: &Client,
        url: &str,
        body: &Value,
        request_timeout: Option<std::time::Duration>,
    ) -> Result<(Value, u16)> {
        let mut req = client.post(url).json(body);
        if let Some(timeout) = request_timeout {
            req = req.timeout(timeout);
        }
        let resp = req.send().await.map_err(|e| {
            PipelineError::Other(format!("Failed to connect to LLM at {}: {}", url, e))
        })?;

        let status = resp.status().as_u16();

        if !resp.status().is_success() {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(Self::parse_retry_after);
            let text = resp.text().await.unwrap_or_default();
            return Err(PipelineError::HttpError {
                status,
                body: text,
                retry_after,
            });
        }

        let json_resp: Value = resp.json().await?;
        Ok((json_resp, status))
    }

    /// Extract metadata fields from an Ollama response.
    fn extract_metadata(json_resp: &Value) -> Option<Value> {
        let mut meta = serde_json::Map::new();
        if let Some(v) = json_resp.get("total_duration") {
            meta.insert("total_duration".into(), v.clone());
        }
        if let Some(v) = json_resp.get("eval_count") {
            meta.insert("eval_count".into(), v.clone());
        }
        if let Some(v) = json_resp.get("eval_duration") {
            meta.insert("eval_duration".into(), v.clone());
        }
        if let Some(v) = json_resp.get("prompt_eval_count") {
            meta.insert("prompt_eval_count".into(), v.clone());
        }
        if let Some(v) = json_resp.get("model") {
            meta.insert("model".into(), v.clone());
        }
        if meta.is_empty() {
            None
        } else {
            Some(Value::Object(meta))
        }
    }

    /// Extract normalized token usage from an Ollama response.
    fn extract_token_usage(json_resp: &Value) -> Option<TokenUsage> {
        let prompt_tokens = json_resp
            .get("prompt_eval_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let completion_tokens = json_resp
            .get("eval_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        if prompt_tokens == 0 && completion_tokens == 0 {
            return None;
        }
        Some(TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            cache_read_tokens: None,
            cache_write_tokens: None,
        })
    }

    /// Extract finish reason from an Ollama response, if reported.
    fn extract_finish_reason(json_resp: &Value) -> Option<String> {
        json_resp
            .get("done_reason")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }
}

#[async_trait]
impl Backend for OllamaBackend {
    async fn complete(
        &self,
        client: &Client,
        base_url: &str,
        request: &LlmRequest,
    ) -> Result<LlmResponse> {
        let base = base_url.trim_end_matches('/');

        if Self::use_chat(request) {
            // Chat endpoint
            let body = Self::build_chat_body(request, false);
            let url = format!("{}/api/chat", base);
            let (json_resp, status) =
                Self::send_request(client, &url, &body, request.request_timeout).await?;

            let text = json_resp
                .get("message")
                .and_then(|m| m.get("content"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            Ok(LlmResponse {
                text,
                status,
                metadata: Self::extract_metadata(&json_resp),
                provider_meta: ProviderMeta::default(),
                token_usage: Self::extract_token_usage(&json_resp),
                finish_reason: Self::extract_finish_reason(&json_resp),
                ttft_ms: None,
                cache_hit: false,
            })
        } else {
            // Generate endpoint
            let body = Self::build_generate_body(request, false);
            let url = format!("{}/api/generate", base);
            let (json_resp, status) =
                Self::send_request(client, &url, &body, request.request_timeout).await?;

            let text = json_resp
                .get("response")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            Ok(LlmResponse {
                text,
                status,
                metadata: Self::extract_metadata(&json_resp),
                provider_meta: ProviderMeta::default(),
                token_usage: Self::extract_token_usage(&json_resp),
                finish_reason: Self::extract_finish_reason(&json_resp),
                ttft_ms: None,
                cache_hit: false,
            })
        }
    }

    async fn complete_streaming(
        &self,
        client: &Client,
        base_url: &str,
        request: &LlmRequest,
        on_token: &mut (dyn FnMut(String) + Send),
    ) -> Result<LlmResponse> {
        let base = base_url.trim_end_matches('/');
        let use_chat = Self::use_chat(request);

        let (url, body) = if use_chat {
            (
                format!("{}/api/chat", base),
                Self::build_chat_body(request, true),
            )
        } else {
            (
                format!("{}/api/generate", base),
                Self::build_generate_body(request, true),
            )
        };

        let mut req = client.post(&url).json(&body);
        if let Some(timeout) = request.request_timeout {
            req = req.timeout(timeout);
        }
        let resp = req.send().await.map_err(|e| {
            PipelineError::Other(format!("Failed to connect to LLM at {}: {}", url, e))
        })?;

        let status = resp.status().as_u16();

        if !resp.status().is_success() {
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(Self::parse_retry_after);
            let text = resp.text().await.unwrap_or_default();
            return Err(PipelineError::HttpError {
                status,
                body: text,
                retry_after,
            });
        }

        let mut stream = resp.bytes_stream();
        let mut decoder = StreamingDecoder::new();
        let mut accumulated = String::new();
        let mut last_metadata = None;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(PipelineError::Request)?;
            for json_val in decoder.decode(&chunk) {
                let token_str = if use_chat {
                    json_val
                        .get("message")
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_str())
                } else {
                    json_val.get("response").and_then(|r| r.as_str())
                };
                if let Some(t) = token_str {
                    if !t.is_empty() {
                        accumulated.push_str(t);
                        on_token(t.to_string());
                    }
                }
                if json_val.get("done").and_then(|v| v.as_bool()) == Some(true) {
                    last_metadata = Self::extract_metadata(&json_val);
                }
            }
        }

        // Flush remaining buffer
        if let Some(json_val) = decoder.flush() {
            let token_str = if use_chat {
                json_val
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
            } else {
                json_val.get("response").and_then(|r| r.as_str())
            };
            if let Some(t) = token_str {
                if !t.is_empty() {
                    accumulated.push_str(t);
                    on_token(t.to_string());
                }
            }
            if json_val.get("done").and_then(|v| v.as_bool()) == Some(true) {
                last_metadata = Self::extract_metadata(&json_val);
            }
        }

        let usage_metadata = last_metadata.clone();
        Ok(LlmResponse {
            text: accumulated,
            status,
            metadata: last_metadata,
            provider_meta: ProviderMeta::default(),
            token_usage: usage_metadata.as_ref().and_then(Self::extract_token_usage),
            finish_reason: usage_metadata
                .as_ref()
                .and_then(Self::extract_finish_reason),
            ttft_ms: None,
            cache_hit: false,
        })
    }

    fn name(&self) -> &'static str {
        "ollama"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{ChatMessage, Role};
    use crate::client::LlmConfig;

    fn test_request() -> LlmRequest {
        LlmRequest {
            model: "llama3.2".into(),
            system_prompt: None,
            prompt: "Why is the sky blue?".into(),
            messages: Vec::new(),
            config: LlmConfig::default(),
            constraint: crate::GenerationConstraint::default(),
            max_tokens_limit: None,
            stream: false,
            request_timeout: None,
        }
    }

    #[test]
    fn test_ollama_backend_generate_payload() {
        let request = test_request();
        let body = OllamaBackend::build_generate_body(&request, false);

        assert_eq!(body["model"], "llama3.2");
        assert_eq!(body["prompt"], "Why is the sky blue?");
        assert_eq!(body["stream"], false);
        assert_eq!(body["options"]["temperature"], 0.7);
        assert_eq!(body["options"]["num_predict"], 2048);
        // No format field when json_mode is false
        assert!(body.get("format").is_none());
    }

    #[test]
    fn test_ollama_backend_chat_payload() {
        let mut request = test_request();
        request.system_prompt = Some("You are a helpful assistant.".into());

        let body = OllamaBackend::build_chat_body(&request, false);

        assert_eq!(body["model"], "llama3.2");
        assert_eq!(body["stream"], false);

        let messages = match body["messages"].as_array() {
            Some(messages) => messages,
            None => panic!("messages array"),
        };
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are a helpful assistant.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "Why is the sky blue?");
    }

    #[test]
    fn test_ollama_backend_json_mode() {
        let mut request = test_request();
        request.config.json_mode = true;

        let body = OllamaBackend::build_generate_body(&request, false);
        assert_eq!(body["format"], "json");

        let chat_body = OllamaBackend::build_chat_body(&request, false);
        assert_eq!(chat_body["format"], "json");
    }

    #[test]
    fn test_ollama_backend_use_chat_logic() {
        let mut request = test_request();

        // No system prompt, no messages → generate
        assert!(!OllamaBackend::use_chat(&request));

        // With system prompt → chat
        request.system_prompt = Some("You are helpful.".into());
        assert!(OllamaBackend::use_chat(&request));

        // Empty system prompt → generate
        request.system_prompt = Some(String::new());
        assert!(!OllamaBackend::use_chat(&request));

        // With messages → chat
        request.system_prompt = None;
        request.messages.push(ChatMessage {
            role: Role::User,
            content: "hello".into(),
        });
        assert!(OllamaBackend::use_chat(&request));
    }

    #[test]
    fn test_ollama_backend_thinking_mode() {
        let mut request = test_request();
        request.config.thinking = true;

        let body = OllamaBackend::build_generate_body(&request, false);
        assert_eq!(body["options"]["extended_thinking"], true);
    }

    #[test]
    fn test_ollama_backend_custom_options() {
        let mut request = test_request();
        request.config.options = Some(json!({"top_p": 0.9, "seed": 42}));

        let body = OllamaBackend::build_generate_body(&request, false);
        assert_eq!(body["options"]["top_p"], 0.9);
        assert_eq!(body["options"]["seed"], 42);
        // Standard options still present
        assert_eq!(body["options"]["temperature"], 0.7);
    }

    #[test]
    fn test_ollama_backend_chat_with_history() {
        let mut request = test_request();
        request.system_prompt = Some("Be helpful.".into());
        request.messages = vec![
            ChatMessage {
                role: Role::User,
                content: "What is 2+2?".into(),
            },
            ChatMessage {
                role: Role::Assistant,
                content: "4".into(),
            },
            ChatMessage {
                role: Role::User,
                content: "And 3+3?".into(),
            },
        ];

        let body = OllamaBackend::build_chat_body(&request, false);
        let messages = match body["messages"].as_array() {
            Some(messages) => messages,
            None => panic!("messages"),
        };
        // system + 3 history messages (no extra user message since messages is non-empty)
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "What is 2+2?");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[3]["role"], "user");
        assert_eq!(messages[3]["content"], "And 3+3?");
    }

    #[test]
    fn test_ollama_backend_streaming_body() {
        let request = test_request();
        let body = OllamaBackend::build_generate_body(&request, true);
        assert_eq!(body["stream"], true);
    }
}
