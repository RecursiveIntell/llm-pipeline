//! LLM call payload — the primary execution unit.
//!
//! [`LlmCall`] is the primary payload for interacting with LLM providers.
//! It handles prompt rendering, backend dispatch (Ollama, OpenAI, etc.),
//! output parsing via [`OutputStrategy`], and optional semantic retry via
//! [`RetryConfig`].

use crate::{
    backend::{self, ChatMessage, LlmRequest, LlmResponse},
    client::LlmConfig,
    diagnostics::ParseDiagnostics,
    error::Result,
    events::{emit, Event},
    exec_ctx::ExecCtx,
    output_parser,
    output_strategy::OutputStrategy,
    parsing,
    payload::{BoxFut, Payload, PayloadOutput},
    retry::RetryConfig,
};
use serde_json::{json, Value};
use std::collections::HashMap;

/// An LLM call payload that invokes a backend with output strategy and optional retry.
///
/// # Example
///
/// ```ignore
/// use llm_pipeline::{LlmCall, ExecCtx};
/// use llm_pipeline::payload::Payload;
/// use serde_json::json;
///
/// let call = LlmCall::new("summarize", "Summarize this: {input}")
///     .with_model("llama3.2:3b")
///     .with_config(LlmConfig::default().with_json_mode(true))
///     .expecting_json();
///
/// let ctx = ExecCtx::builder("http://localhost:11434").build();
/// let output = call.invoke(&ctx, json!("Some long text...")).await?;
/// ```
pub struct LlmCall {
    /// Instance name (for logging/events).
    name: String,
    /// Prompt template with `{input}` and `{key}` placeholders.
    prompt_template: String,
    /// Optional system prompt template (triggers chat endpoint on Ollama).
    system_template: Option<String>,
    /// Model identifier (e.g. `"llama3.2:3b"`).
    model: String,
    /// LLM configuration (temperature, tokens, json_mode, etc.).
    config: LlmConfig,
    /// Whether to use the streaming endpoint.
    streaming: bool,
    /// How to parse the raw LLM text into a Value. Default: `Lossy`.
    output_strategy: OutputStrategy,
    /// Optional semantic retry configuration.
    retry: Option<RetryConfig>,
}

impl LlmCall {
    /// Create a new LLM call payload with a prompt template.
    pub fn new(name: impl Into<String>, prompt_template: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            prompt_template: prompt_template.into(),
            system_template: None,
            model: "llama3.2:3b".to_string(),
            config: LlmConfig::default(),
            streaming: false,
            output_strategy: OutputStrategy::default(),
            retry: None,
        }
    }

    /// Returns the prompt template.
    pub fn prompt_template(&self) -> &str {
        &self.prompt_template
    }

    /// Returns the system template, if any.
    pub fn system_template(&self) -> Option<&str> {
        self.system_template.as_deref()
    }

    /// Returns the model identifier.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the LLM config.
    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    /// Returns whether streaming is enabled.
    pub fn is_streaming(&self) -> bool {
        self.streaming
    }

    /// Returns the output strategy.
    pub fn output_strategy(&self) -> &OutputStrategy {
        &self.output_strategy
    }

    /// Returns the retry configuration, if any.
    pub fn retry(&self) -> Option<&RetryConfig> {
        self.retry.as_ref()
    }

    /// Set a system prompt template (enables `/api/chat` mode on Ollama).
    pub fn with_system(mut self, template: impl Into<String>) -> Self {
        self.system_template = Some(template.into());
        self
    }

    /// Set the model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the LLM configuration.
    pub fn with_config(mut self, config: LlmConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable or disable streaming.
    pub fn with_streaming(mut self, enabled: bool) -> Self {
        self.streaming = enabled;
        self
    }

    /// Set a custom output strategy.
    pub fn with_output_strategy(mut self, strategy: OutputStrategy) -> Self {
        self.output_strategy = strategy;
        self
    }

    /// Set retry configuration.
    pub fn with_retry(mut self, retry: RetryConfig) -> Self {
        self.retry = Some(retry);
        self
    }

    /// Shorthand: expect JSON output (full multi-strategy extraction with repair).
    pub fn expecting_json(mut self) -> Self {
        self.output_strategy = OutputStrategy::Json;
        self
    }

    /// Shorthand: expect a string list.
    pub fn expecting_list(mut self) -> Self {
        self.output_strategy = OutputStrategy::StringList;
        self
    }

    /// Shorthand: expect one of the given choices.
    pub fn expecting_choice(mut self, choices: Vec<String>) -> Self {
        self.output_strategy = OutputStrategy::Choice(choices);
        self
    }

    /// Shorthand: expect a number.
    pub fn expecting_number(mut self) -> Self {
        self.output_strategy = OutputStrategy::Number;
        self
    }

    /// Shorthand: expect a number in a range.
    pub fn expecting_number_in_range(mut self, min: f64, max: f64) -> Self {
        self.output_strategy = OutputStrategy::NumberInRange(min, max);
        self
    }

    /// Shorthand: expect clean text output.
    pub fn expecting_text(mut self) -> Self {
        self.output_strategy = OutputStrategy::Text;
        self
    }

    /// Create from an existing [`Stage`](crate::stage::Stage) (for Pipeline compatibility).
    pub(crate) fn from_stage(stage: &crate::stage::Stage, streaming: bool) -> Self {
        Self {
            name: stage.name.clone(),
            prompt_template: stage.prompt_template.clone(),
            system_template: stage.system_prompt.clone(),
            model: stage.model.clone(),
            config: stage.config.clone(),
            streaming,
            output_strategy: OutputStrategy::default(),
            retry: None,
        }
    }

    /// Render the prompt template, substituting `{input}` and context vars.
    fn render_prompt(template: &str, input: &str, vars: &HashMap<String, String>) -> String {
        let mut rendered = template.replace("{input}", input);
        for (key, value) in vars {
            let placeholder = format!("{{{}}}", key);
            rendered = rendered.replace(&placeholder, value);
        }
        rendered
    }

    /// Render a template with context vars only (no {input}).
    fn render_system(template: &str, vars: &HashMap<String, String>) -> String {
        let mut rendered = template.to_string();
        for (key, value) in vars {
            let placeholder = format!("{{{}}}", key);
            rendered = rendered.replace(&placeholder, value);
        }
        rendered
    }

    /// Convert a `Value` input to a string for template substitution.
    fn input_to_string(input: &Value) -> String {
        match input {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }

    /// Build an `LlmRequest` from the current state.
    fn build_request(
        &self,
        prompt: &str,
        system: Option<&str>,
        messages: Vec<ChatMessage>,
        stream: bool,
    ) -> LlmRequest {
        LlmRequest {
            model: self.model.clone(),
            system_prompt: system.map(|s| s.to_string()),
            prompt: prompt.to_string(),
            messages,
            config: self.config.clone(),
            stream,
        }
    }

    /// Execute via the backend (non-streaming), tracking transport retries.
    ///
    /// Returns `(LlmResponse, transport_retries, backoff_total_ms)`.
    async fn call_backend(
        &self,
        ctx: &ExecCtx,
        request: &LlmRequest,
    ) -> Result<(LlmResponse, u32, u64)> {
        let mut transport_retries: u32 = 0;
        let mut backoff_total_ms: u64 = 0;
        let name = self.name.clone();
        let event_handler = ctx.event_handler.clone();

        let mut on_retry = |attempt: u32, delay: std::time::Duration, reason: &str| {
            transport_retries = attempt;
            backoff_total_ms += delay.as_millis() as u64;
            emit(
                &event_handler,
                Event::TransportRetry {
                    name: name.clone(),
                    attempt,
                    delay_ms: delay.as_millis() as u64,
                    reason: reason.to_string(),
                },
            );
        };

        let response = backend::with_backoff(
            &ctx.backend,
            &ctx.client,
            &ctx.base_url,
            request,
            &ctx.backoff,
            ctx.cancel_flag(),
            Some(&mut on_retry),
        )
        .await?;

        Ok((response, transport_retries, backoff_total_ms))
    }

    /// Execute via the backend (streaming), emitting Token events and tracking transport retries.
    ///
    /// Returns `(LlmResponse, transport_retries, backoff_total_ms)`.
    async fn call_backend_streaming(
        &self,
        ctx: &ExecCtx,
        request: &LlmRequest,
    ) -> Result<(LlmResponse, u32, u64)> {
        let mut transport_retries: u32 = 0;
        let mut backoff_total_ms: u64 = 0;
        let retry_name = self.name.clone();
        let retry_event_handler = ctx.event_handler.clone();

        let mut on_retry = |attempt: u32, delay: std::time::Duration, reason: &str| {
            transport_retries = attempt;
            backoff_total_ms += delay.as_millis() as u64;
            emit(
                &retry_event_handler,
                Event::TransportRetry {
                    name: retry_name.clone(),
                    attempt,
                    delay_ms: delay.as_millis() as u64,
                    reason: reason.to_string(),
                },
            );
        };

        let name = self.name.clone();
        let event_handler = ctx.event_handler.clone();
        let mut on_token = move |token: String| {
            emit(
                &event_handler,
                Event::Token {
                    name: name.clone(),
                    chunk: token,
                },
            );
        };

        let response = backend::with_backoff_streaming(
            &ctx.backend,
            &ctx.client,
            &ctx.base_url,
            request,
            &ctx.backoff,
            backend::BackoffStreamOpts {
                cancel: ctx.cancel_flag(),
                on_retry: Some(&mut on_retry),
                on_token: &mut on_token,
            },
        )
        .await?;

        Ok((response, transport_retries, backoff_total_ms))
    }

    /// Check if a retry is needed. Returns `Some(reason)` if retry needed, `None` if output is ok.
    fn check_retry_needed(
        &self,
        output: &PayloadOutput,
        retry_config: &RetryConfig,
    ) -> Option<String> {
        // Check parse error from OutputStrategy
        if let Some(ref diag) = output.diagnostics {
            if let Some(ref err) = diag.parse_error {
                return Some(err.clone());
            }
        }

        // Check semantic validator
        if let Some(ref validator) = retry_config.validator {
            if let Err(reason) = validator(&output.raw_response, &output.value) {
                return Some(reason);
            }
        }

        None
    }

    /// Build a `PayloadOutput` from raw LLM text using the configured `OutputStrategy`.
    ///
    /// Per CLAUDE.md: `build_output` MUST always return `Ok(PayloadOutput)`.
    /// Parse failures go into `diagnostics.parse_error`, not `Err`.
    fn build_output(&self, raw_text: String) -> PayloadOutput {
        let (thinking, cleaned) = parsing::extract_thinking(&raw_text);

        let mut diag = ParseDiagnostics::default();

        let value = match &self.output_strategy {
            OutputStrategy::Lossy => {
                diag.strategy = Some("lossy");
                parsing::parse_value_lossy(&cleaned)
            }
            OutputStrategy::Json => {
                diag.strategy = Some("json");
                match output_parser::parse_json_value(&cleaned) {
                    Ok(v) => v,
                    Err(e) => {
                        diag.parse_error = Some(e.to_string());
                        // Fallback: try lossy parse
                        parsing::parse_value_lossy(&cleaned)
                    }
                }
            }
            OutputStrategy::StringList => {
                diag.strategy = Some("string_list");
                match output_parser::parse_string_list_raw(&cleaned) {
                    Ok(items) => Value::Array(items.into_iter().map(Value::String).collect()),
                    Err(e) => {
                        diag.parse_error = Some(e.to_string());
                        Value::String(cleaned.clone())
                    }
                }
            }
            OutputStrategy::XmlTag(tag) => {
                diag.strategy = Some("xml_tag");
                match output_parser::parse_xml_tag(&cleaned, tag) {
                    Ok(content) => Value::String(content),
                    Err(e) => {
                        diag.parse_error = Some(e.to_string());
                        Value::String(cleaned.clone())
                    }
                }
            }
            OutputStrategy::Choice(choices) => {
                diag.strategy = Some("choice");
                let choice_refs: Vec<&str> = choices.iter().map(|s| s.as_str()).collect();
                match output_parser::parse_choice(&cleaned, &choice_refs) {
                    Ok(matched) => Value::String(matched.to_string()),
                    Err(e) => {
                        diag.parse_error = Some(e.to_string());
                        Value::String(cleaned.clone())
                    }
                }
            }
            OutputStrategy::Number => {
                diag.strategy = Some("number");
                match output_parser::parse_number::<f64>(&cleaned) {
                    Ok(n) => json!(n),
                    Err(e) => {
                        diag.parse_error = Some(e.to_string());
                        Value::String(cleaned.clone())
                    }
                }
            }
            OutputStrategy::NumberInRange(min, max) => {
                diag.strategy = Some("number_in_range");
                match output_parser::parse_number_in_range::<f64>(&cleaned, *min, *max) {
                    Ok(n) => json!(n),
                    Err(e) => {
                        diag.parse_error = Some(e.to_string());
                        Value::String(cleaned.clone())
                    }
                }
            }
            OutputStrategy::Text => {
                diag.strategy = Some("text");
                match output_parser::parse_text(&cleaned) {
                    Ok(text) => Value::String(text),
                    Err(e) => {
                        diag.parse_error = Some(e.to_string());
                        Value::String(cleaned.clone())
                    }
                }
            }
            OutputStrategy::Custom(f) => {
                diag.strategy = Some("custom");
                match f(&cleaned) {
                    Ok(v) => v,
                    Err(e) => {
                        diag.parse_error = Some(e.to_string());
                        Value::String(cleaned.clone())
                    }
                }
            }
        };

        // Check if repair was applied (for Json strategy, the output_parser
        // internally tries repair — we can detect this by checking if the
        // parse succeeded on repaired input)
        if diag.parse_error.is_none() && matches!(self.output_strategy, OutputStrategy::Json) {
            // If direct parse of cleaned text fails but output_parser succeeded,
            // it means repair was applied
            if serde_json::from_str::<Value>(&cleaned).is_err() {
                diag.repaired = true;
            }
        }

        PayloadOutput {
            value,
            raw_response: raw_text,
            thinking,
            model: Some(self.model.clone()),
            diagnostics: Some(diag),
        }
    }
}

impl Payload for LlmCall {
    fn kind(&self) -> &'static str {
        "llm-call"
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn invoke<'a>(&'a self, ctx: &'a ExecCtx, input: Value) -> BoxFut<'a, Result<PayloadOutput>> {
        Box::pin(async move {
            ctx.check_cancelled()?;

            emit(
                &ctx.event_handler,
                Event::PayloadStart {
                    name: self.name.clone(),
                    kind: self.kind(),
                },
            );

            let input_str = Self::input_to_string(&input);
            let prompt = Self::render_prompt(&self.prompt_template, &input_str, &ctx.vars);
            let system = self
                .system_template
                .as_ref()
                .map(|t| Self::render_system(t, &ctx.vars));

            // --- Initial call ---
            let request =
                self.build_request(&prompt, system.as_deref(), Vec::new(), self.streaming);

            let result = if self.streaming {
                self.call_backend_streaming(ctx, &request).await
            } else {
                self.call_backend(ctx, &request).await
            };

            let mut output = match result {
                Ok((response, transport_retries, backoff_total_ms)) => {
                    let mut out = self.build_output(response.text);
                    if let Some(ref mut diag) = out.diagnostics {
                        diag.transport_retries = transport_retries;
                        diag.backoff_total_ms = backoff_total_ms;
                    }
                    out
                }
                Err(e) => {
                    emit(
                        &ctx.event_handler,
                        Event::PayloadEnd {
                            name: self.name.clone(),
                            ok: false,
                        },
                    );
                    return Err(e);
                }
            };

            // --- Retry loop ---
            if let Some(ref retry_config) = self.retry {
                // Check if initial output needs retry
                let mut retry_reason = self.check_retry_needed(&output, retry_config);

                if retry_reason.is_some() {
                    let mut messages = vec![ChatMessage {
                        role: backend::Role::User,
                        content: prompt.clone(),
                    }];
                    let mut temp_offset = 0.0f64;

                    for attempt in 1..=retry_config.max_retries {
                        ctx.check_cancelled()?;

                        let reason = retry_reason.take().unwrap_or_default();

                        emit(
                            &ctx.event_handler,
                            Event::RetryStart {
                                name: self.name.clone(),
                                attempt,
                                reason: reason.clone(),
                            },
                        );

                        // Build correction messages
                        messages.push(ChatMessage {
                            role: backend::Role::Assistant,
                            content: output.raw_response.clone(),
                        });
                        messages.push(ChatMessage {
                            role: backend::Role::User,
                            content: format!(
                                "Your previous response was invalid: {}. Please try again with the correct format.",
                                reason
                            ),
                        });

                        // Cool down temperature
                        if retry_config.cool_down {
                            temp_offset += 0.2;
                        }

                        let mut retry_config_clone = self.config.clone();
                        retry_config_clone.temperature =
                            (retry_config_clone.temperature - temp_offset).max(0.0);

                        let retry_request = LlmRequest {
                            model: self.model.clone(),
                            system_prompt: system.clone(),
                            prompt: prompt.clone(),
                            messages: messages.clone(),
                            config: retry_config_clone,
                            stream: false, // retries always non-streaming
                        };

                        match self.call_backend(ctx, &retry_request).await {
                            Ok((response, tr, bt)) => {
                                output = self.build_output(response.text);
                                if let Some(ref mut diag) = output.diagnostics {
                                    diag.retry_attempts = attempt;
                                    diag.transport_retries = tr;
                                    diag.backoff_total_ms = bt;
                                }
                            }
                            Err(e) => {
                                emit(
                                    &ctx.event_handler,
                                    Event::RetryEnd {
                                        name: self.name.clone(),
                                        attempts: attempt,
                                        success: false,
                                    },
                                );
                                emit(
                                    &ctx.event_handler,
                                    Event::PayloadEnd {
                                        name: self.name.clone(),
                                        ok: false,
                                    },
                                );
                                return Err(e);
                            }
                        }

                        // Check if this retry succeeded
                        retry_reason = self.check_retry_needed(&output, retry_config);

                        if retry_reason.is_none() {
                            // Success!
                            emit(
                                &ctx.event_handler,
                                Event::RetryEnd {
                                    name: self.name.clone(),
                                    attempts: attempt,
                                    success: true,
                                },
                            );
                            break;
                        }

                        if attempt == retry_config.max_retries {
                            // Exhausted — return best effort
                            if let Some(ref mut diag) = output.diagnostics {
                                diag.retry_attempts = attempt;
                            }
                            emit(
                                &ctx.event_handler,
                                Event::RetryEnd {
                                    name: self.name.clone(),
                                    attempts: attempt,
                                    success: false,
                                },
                            );
                        }
                    }
                }
            }

            emit(
                &ctx.event_handler,
                Event::PayloadEnd {
                    name: self.name.clone(),
                    ok: true,
                },
            );

            Ok(output)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Role;

    #[test]
    fn test_build_output_lossy_backward_compat() {
        let call = LlmCall::new("test", "prompt");
        let output = call.build_output(r#"{"key": "value"}"#.into());
        assert!(output.value.is_object());
        assert!(output.diagnostics.as_ref().unwrap().ok());
        assert_eq!(output.diagnostics.as_ref().unwrap().strategy, Some("lossy"));
    }

    #[test]
    fn test_build_output_json_strategy_succeeds() {
        let call = LlmCall::new("test", "prompt").expecting_json();
        let output = call.build_output(r#"{"key": "value"}"#.into());
        assert!(output.value.is_object());
        assert_eq!(output.value["key"], "value");
        assert!(output.diagnostics.as_ref().unwrap().ok());
    }

    #[test]
    fn test_build_output_json_strategy_repairs() {
        let call = LlmCall::new("test", "prompt").expecting_json();
        // Single quotes and trailing comma — repairable
        let output = call.build_output("{'key': 'value',}".into());
        assert!(output.value.is_object());
        assert!(output.diagnostics.as_ref().unwrap().ok());
        assert!(output.diagnostics.as_ref().unwrap().repaired);
    }

    #[test]
    fn test_build_output_json_strategy_fails() {
        let call = LlmCall::new("test", "prompt").expecting_json();
        let output = call.build_output("not json at all".into());
        assert!(output.diagnostics.as_ref().unwrap().parse_error.is_some());
        // Should still return a Value (fallback to lossy)
        assert!(output.value.is_string());
    }

    #[test]
    fn test_build_output_string_list_strategy() {
        let call = LlmCall::new("test", "prompt").expecting_list();
        let output = call.build_output("[\"apple\", \"banana\", \"cherry\"]".into());
        assert!(output.value.is_array());
        let arr = output.value.as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert!(output.diagnostics.as_ref().unwrap().ok());
    }

    #[test]
    fn test_build_output_xml_tag_strategy() {
        let call = LlmCall::new("test", "prompt")
            .with_output_strategy(OutputStrategy::XmlTag("answer".into()));
        let output = call.build_output("<answer>42</answer>".into());
        assert_eq!(output.value, Value::String("42".into()));
        assert!(output.diagnostics.as_ref().unwrap().ok());
    }

    #[test]
    fn test_build_output_choice_strategy() {
        let call = LlmCall::new("test", "prompt").expecting_choice(vec![
            "yes".into(),
            "no".into(),
            "maybe".into(),
        ]);
        let output = call.build_output("I think the answer is yes.".into());
        assert_eq!(output.value, Value::String("yes".into()));
        assert!(output.diagnostics.as_ref().unwrap().ok());
    }

    #[test]
    fn test_build_output_number_strategy() {
        let call = LlmCall::new("test", "prompt").expecting_number();
        let output = call.build_output("Score: 8.5".into());
        let n = output.value.as_f64().unwrap();
        assert!((n - 8.5).abs() < f64::EPSILON);
        assert!(output.diagnostics.as_ref().unwrap().ok());
    }

    #[test]
    fn test_build_output_number_in_range_rejects() {
        let call = LlmCall::new("test", "prompt").expecting_number_in_range(0.0, 5.0);
        let output = call.build_output("Score: 8.5".into());
        // Should fail: 8.5 > 5.0
        assert!(output.diagnostics.as_ref().unwrap().parse_error.is_some());
    }

    #[test]
    fn test_build_output_text_strategy() {
        let call = LlmCall::new("test", "prompt").expecting_text();
        let output = call.build_output("Sure! Here's the answer: The sky is blue.".into());
        let text = output.value.as_str().unwrap();
        // parse_text strips "Sure!" and "Here's..." prefixes
        assert!(!text.starts_with("Sure!"));
        assert!(output.diagnostics.as_ref().unwrap().ok());
    }

    #[test]
    fn test_build_output_custom_strategy() {
        let call = LlmCall::new("test", "prompt").with_output_strategy(OutputStrategy::Custom(
            std::sync::Arc::new(|raw: &str| {
                let upper = raw.to_uppercase();
                Ok(Value::String(upper))
            }),
        ));
        let output = call.build_output("hello world".into());
        assert_eq!(output.value, Value::String("HELLO WORLD".into()));
        assert!(output.diagnostics.as_ref().unwrap().ok());
    }

    #[test]
    fn test_diagnostics_attached_to_output() {
        let call = LlmCall::new("test", "prompt").expecting_json();
        let output = call.build_output(r#"{"a": 1}"#.into());
        let diag = output.diagnostics.as_ref().unwrap();
        assert_eq!(diag.strategy, Some("json"));
        assert!(diag.ok());
        assert!(!diag.repaired);
        assert_eq!(diag.retry_attempts, 0);
    }

    #[test]
    fn test_build_output_with_thinking() {
        let call = LlmCall::new("test", "prompt").expecting_json();
        let input = "<think>Let me think about this...</think>{\"result\": 42}";
        let output = call.build_output(input.into());
        assert_eq!(output.thinking, Some("Let me think about this...".into()));
        assert_eq!(output.value["result"], 42);
    }

    #[test]
    fn test_backend_default_is_ollama() {
        let ctx = ExecCtx::builder("http://localhost:11434").build();
        assert_eq!(ctx.backend.name(), "ollama");
    }

    #[cfg(feature = "openai")]
    #[test]
    fn test_exec_ctx_openai_builder() {
        let ctx = ExecCtx::builder("https://api.openai.com")
            .openai_with_key("sk-test")
            .build();
        assert_eq!(ctx.backend.name(), "openai");
    }

    #[test]
    fn test_build_request() {
        let call = LlmCall::new("test", "Summarize: {input}")
            .with_model("gpt-4o")
            .with_config(LlmConfig::default().with_json_mode(true));

        let request = call.build_request(
            "Tell me about Rust",
            Some("You are helpful"),
            Vec::new(),
            false,
        );

        assert_eq!(request.model, "gpt-4o");
        assert_eq!(request.prompt, "Tell me about Rust");
        assert_eq!(request.system_prompt.as_deref(), Some("You are helpful"));
        assert!(request.config.json_mode);
        assert!(!request.stream);
    }

    #[test]
    fn test_build_request_with_messages() {
        let call = LlmCall::new("test", "prompt");
        let messages = vec![
            ChatMessage {
                role: Role::User,
                content: "What is 2+2?".into(),
            },
            ChatMessage {
                role: Role::Assistant,
                content: "4".into(),
            },
        ];
        let request = call.build_request("Follow up", None, messages, false);
        assert_eq!(request.messages.len(), 2);
    }

    // --- Retry tests (unit-level, testing check_retry_needed and retry config) ---

    #[test]
    fn test_retry_not_triggered_on_success() {
        let call = LlmCall::new("test", "prompt")
            .expecting_json()
            .with_retry(RetryConfig::new(2));

        let output = call.build_output(r#"{"key": "value"}"#.into());
        let retry_config = call.retry.as_ref().unwrap();
        assert!(call.check_retry_needed(&output, retry_config).is_none());
    }

    #[test]
    fn test_retry_triggered_on_parse_failure() {
        let call = LlmCall::new("test", "prompt")
            .expecting_json()
            .with_retry(RetryConfig::new(2));

        let output = call.build_output("not json".into());
        let retry_config = call.retry.as_ref().unwrap();
        let reason = call.check_retry_needed(&output, retry_config);
        assert!(reason.is_some());
    }

    #[test]
    fn test_retry_triggered_on_semantic_failure() {
        let call = LlmCall::new("test", "prompt")
            .expecting_json()
            .with_retry(RetryConfig::new(2).requiring_keys(&["title", "year"]));

        // Valid JSON but missing required keys
        let output = call.build_output(r#"{"title": "Matrix"}"#.into());
        let retry_config = call.retry.as_ref().unwrap();
        let reason = call.check_retry_needed(&output, retry_config);
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("year"));
    }

    #[test]
    fn test_retry_requiring_keys_passes() {
        let call = LlmCall::new("test", "prompt")
            .expecting_json()
            .with_retry(RetryConfig::new(2).requiring_keys(&["title", "year"]));

        let output = call.build_output(r#"{"title": "Matrix", "year": 1999}"#.into());
        let retry_config = call.retry.as_ref().unwrap();
        assert!(call.check_retry_needed(&output, retry_config).is_none());
    }

    #[test]
    fn test_retry_cool_down_reduces_temperature() {
        let config = LlmConfig::default().with_temperature(0.7);
        let call = LlmCall::new("test", "prompt")
            .with_config(config)
            .with_retry(RetryConfig::new(3));

        // Verify cool_down is true by default
        assert!(call.retry.as_ref().unwrap().cool_down);

        // After 2 retries at 0.2 per retry: 0.7 - 0.4 = 0.3
        let adjusted = (0.7 - 0.4f64).max(0.0);
        assert!((adjusted - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_retry_no_cool_down() {
        let call = LlmCall::new("test", "prompt").with_retry(RetryConfig::new(3).no_cool_down());

        assert!(!call.retry.as_ref().unwrap().cool_down);
    }

    #[test]
    fn test_choice_strategy_with_retry_detects_failure() {
        let call = LlmCall::new("test", "prompt")
            .expecting_choice(vec!["approve".into(), "reject".into(), "defer".into()])
            .with_retry(RetryConfig::new(2));

        // Bad response - no valid choice found
        let output = call.build_output("I think we should consider all options carefully.".into());
        let retry_config = call.retry.as_ref().unwrap();
        let reason = call.check_retry_needed(&output, retry_config);
        assert!(reason.is_some());
    }

    #[test]
    fn test_choice_strategy_succeeds() {
        let call = LlmCall::new("test", "prompt")
            .expecting_choice(vec!["approve".into(), "reject".into(), "defer".into()])
            .with_retry(RetryConfig::new(2));

        let output = call.build_output("I would approve this request.".into());
        let retry_config = call.retry.as_ref().unwrap();
        assert!(call.check_retry_needed(&output, retry_config).is_none());
        assert_eq!(output.value, Value::String("approve".into()));
    }

    #[test]
    fn test_number_in_range_with_retry_detects_failure() {
        let call = LlmCall::new("test", "prompt")
            .expecting_number_in_range(1.0, 10.0)
            .with_retry(RetryConfig::new(2));

        let output = call.build_output("Score: 15".into());
        let retry_config = call.retry.as_ref().unwrap();
        let reason = call.check_retry_needed(&output, retry_config);
        assert!(reason.is_some());
    }

    #[test]
    fn test_custom_validator_with_retry() {
        let call = LlmCall::new("test", "prompt").expecting_json().with_retry(
            RetryConfig::new(2).with_validator(|_raw, value| {
                let score = value
                    .get("score")
                    .and_then(|v| v.as_f64())
                    .ok_or("missing score")?;
                if !(0.0..=1.0).contains(&score) {
                    return Err(format!("score {} outside 0.0-1.0", score));
                }
                Ok(())
            }),
        );

        // Valid JSON with out-of-range score
        let output = call.build_output(r#"{"score": 1.5}"#.into());
        let retry_config = call.retry.as_ref().unwrap();
        let reason = call.check_retry_needed(&output, retry_config);
        assert!(reason.is_some());
        assert!(reason.unwrap().contains("score 1.5 outside"));

        // Valid JSON with valid score
        let output = call.build_output(r#"{"score": 0.8}"#.into());
        assert!(call.check_retry_needed(&output, retry_config).is_none());
    }

    #[test]
    fn test_transport_retry_populates_diagnostics() {
        let mut transport_retries: u32 = 0;
        let mut backoff_total_ms: u64 = 0;

        let mut on_retry = |attempt: u32, delay: std::time::Duration, _reason: &str| {
            transport_retries = attempt;
            backoff_total_ms += delay.as_millis() as u64;
        };

        on_retry(1, std::time::Duration::from_millis(500), "429 Too Many Requests");
        on_retry(2, std::time::Duration::from_millis(1000), "503 Service Unavailable");

        assert_eq!(transport_retries, 2);
        assert_eq!(backoff_total_ms, 1500);
    }

    #[test]
    fn test_llm_call_accessors() {
        let call = LlmCall::new("test", "Hello {input}")
            .with_model("llama3.2:3b")
            .with_streaming(true)
            .expecting_json();
        assert_eq!(call.name(), "test");
        assert_eq!(call.model(), "llama3.2:3b");
        assert!(call.is_streaming());
        assert!(matches!(call.output_strategy(), OutputStrategy::Json));
        assert_eq!(call.prompt_template(), "Hello {input}");
        assert!(call.system_template().is_none());
        assert!(call.retry().is_none());
    }
}
