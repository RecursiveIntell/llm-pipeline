use crate::{
    error::Result,
    exec_ctx::ExecCtx,
    llm_call::LlmCall,
    parsing,
    payload::Payload,
    stage::Stage,
    streaming::StreamingDecoder,
    types::{PipelineContext, PipelineInput, PipelineProgress, PipelineResult, StageOutput},
    PipelineError,
};
use futures::StreamExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// Pipeline executor for multi-stage LLM workflows.
///
/// Internally converts stages to [`LlmCall`] payloads and executes them
/// sequentially. For new code, consider using [`LlmCall`] + [`Chain`](crate::Chain)
/// directly for more flexibility.
///
/// Pipeline forces all stage outputs to deserialize into the same `T`.
/// If your workflow produces heterogeneous outputs, use the payload API instead.
pub struct Pipeline<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Clone,
{
    stages: Vec<Stage>,
    context: PipelineContext,
    cancellation: Option<Arc<AtomicBool>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> std::fmt::Debug for Pipeline<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field(
                "stages",
                &self.stages.iter().map(|s| &s.name).collect::<Vec<_>>(),
            )
            .field(
                "context_keys",
                &self.context.data.keys().collect::<Vec<_>>(),
            )
            .field("has_cancellation", &self.cancellation.is_some())
            .finish()
    }
}

impl<T> Pipeline<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Clone,
{
    /// Create a new pipeline builder.
    pub fn builder() -> PipelineBuilder<T> {
        PipelineBuilder::new()
    }

    /// Get a reference to the pipeline's stages.
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Check whether cancellation has been requested.
    fn check_cancelled(&self) -> Result<()> {
        if let Some(ref cancel) = self.cancellation {
            if cancel.load(Ordering::Relaxed) {
                return Err(PipelineError::Cancelled);
            }
        }
        Ok(())
    }

    /// Build an `ExecCtx` from pipeline state.
    fn build_ctx(&self, client: &Client, endpoint: &str) -> ExecCtx {
        ExecCtx::builder(endpoint)
            .client(client.clone())
            .vars(self.context.data.clone())
            .cancellation(self.cancellation.clone())
            .build()
    }

    /// Convert enabled stages to LlmCall payloads, returning (stage_index, payload) pairs.
    fn build_payloads(&self, streaming: bool) -> Vec<(usize, LlmCall)> {
        self.stages
            .iter()
            .enumerate()
            .filter(|(_, s)| s.enabled)
            .map(|(idx, stage)| (idx, LlmCall::from_stage(stage, streaming)))
            .collect()
    }

    /// Execute the pipeline in non-streaming mode.
    ///
    /// Each enabled stage runs sequentially. The output of each stage is
    /// serialized to JSON and used as input for the next stage's prompt.
    ///
    /// Internally converts stages to [`LlmCall`] payloads.
    pub async fn execute(
        &self,
        client: &Client,
        endpoint: &str,
        input: PipelineInput,
    ) -> Result<PipelineResult<T>> {
        self.execute_with_progress(client, endpoint, input, |_| {})
            .await
    }

    /// Execute the pipeline with a progress callback (non-streaming LLM calls).
    ///
    /// The callback is invoked at the start of each stage. Stages are executed
    /// as [`LlmCall`] payloads internally.
    pub async fn execute_with_progress<F>(
        &self,
        client: &Client,
        endpoint: &str,
        input: PipelineInput,
        mut on_progress: F,
    ) -> Result<PipelineResult<T>>
    where
        F: FnMut(PipelineProgress),
    {
        let ctx = self.build_ctx(client, endpoint);
        let payloads = self.build_payloads(false);
        let stages_enabled: Vec<bool> = self.stages.iter().map(|s| s.enabled).collect();
        let total_stages = self.stages.len();

        let mut current_input = Value::String(input.idea);
        let mut stage_results = Vec::new();

        for (idx, payload) in &payloads {
            self.check_cancelled()?;

            on_progress(PipelineProgress {
                stage_index: *idx,
                total_stages,
                stage_name: payload.name().to_string(),
                current_step: None,
                total_steps: None,
            });

            let output = payload.invoke(&ctx, current_input).await.map_err(|e| {
                PipelineError::StageFailed {
                    stage: payload.name().to_string(),
                    message: e.to_string(),
                }
            })?;

            // Parse into T from the structured output value
            let parsed: T = output.parse_as().map_err(|e| PipelineError::StageFailed {
                stage: payload.name().to_string(),
                message: e.to_string(),
            })?;

            current_input = output.value;
            stage_results.push(StageOutput {
                output: parsed,
                thinking: output.thinking,
                raw_response: output.raw_response,
            });
        }

        let final_output = stage_results
            .last()
            .ok_or_else(|| PipelineError::Other("No stages were executed".to_string()))?
            .output
            .clone();

        Ok(PipelineResult {
            final_output,
            stage_results,
            stages_enabled,
        })
    }

    /// Execute the pipeline with streaming LLM calls and per-token callbacks.
    ///
    /// Uses buffered line-framing to correctly handle JSON lines split across
    /// chunk boundaries.
    ///
    /// `on_progress` is called at the start of each stage.
    /// `on_token` is called for each token received from the LLM.
    pub async fn execute_streaming<F, G>(
        &self,
        client: &Client,
        endpoint: &str,
        input: PipelineInput,
        mut on_progress: F,
        mut on_token: G,
    ) -> Result<PipelineResult<T>>
    where
        F: FnMut(PipelineProgress),
        G: FnMut(usize, &str),
    {
        let ctx = self.build_ctx(client, endpoint);
        let payloads = self.build_payloads(false); // don't use payload streaming path
        let stages_enabled: Vec<bool> = self.stages.iter().map(|s| s.enabled).collect();
        let total_stages = self.stages.len();

        let mut current_input = Value::String(input.idea);
        let mut stage_results = Vec::new();

        for (idx, payload) in &payloads {
            self.check_cancelled()?;

            on_progress(PipelineProgress {
                stage_index: *idx,
                total_stages,
                stage_name: payload.name().to_string(),
                current_step: None,
                total_steps: None,
            });

            // For streaming, we call the Ollama API directly with the callback
            let input_str = match &current_input {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };

            let prompt = {
                let mut rendered = payload.prompt_template().replace("{input}", &input_str);
                for (key, value) in &ctx.vars {
                    let placeholder = format!("{{{}}}", key);
                    rendered = rendered.replace(&placeholder, value);
                }
                rendered
            };

            let raw_response = self
                .stream_call(client, endpoint, payload, &prompt, *idx, &mut on_token)
                .await
                .map_err(|e| PipelineError::StageFailed {
                    stage: payload.name().to_string(),
                    message: e.to_string(),
                })?;

            let (thinking, cleaned) = parsing::extract_thinking(&raw_response);
            let parsed: T = crate::output_parser::parse_json(&cleaned).map_err(|e| {
                PipelineError::StageFailed {
                    stage: payload.name().to_string(),
                    message: e.to_string(),
                }
            })?;

            current_input = parsing::parse_value_lossy(&cleaned);
            stage_results.push(StageOutput {
                output: parsed,
                thinking,
                raw_response,
            });
        }

        let final_output = stage_results
            .last()
            .ok_or_else(|| PipelineError::Other("No stages were executed".to_string()))?
            .output
            .clone();

        Ok(PipelineResult {
            final_output,
            stage_results,
            stages_enabled,
        })
    }

    /// Perform a single streaming call to Ollama, using buffered line framing.
    async fn stream_call<G>(
        &self,
        client: &Client,
        endpoint: &str,
        payload: &LlmCall,
        prompt: &str,
        stage_idx: usize,
        on_token: &mut G,
    ) -> Result<String>
    where
        G: FnMut(usize, &str),
    {
        let config = payload.config();
        let mut body = json!({
            "model": payload.model(),
            "prompt": prompt,
            "stream": true,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
            },
        });

        if config.thinking {
            body["options"]["extended_thinking"] = json!(true);
        }

        if config.json_mode {
            body["format"] = json!("json");
        }

        // Merge custom options
        if let Some(ref opts) = config.options {
            if let Some(options) = body["options"].as_object_mut() {
                if let Some(custom) = opts.as_object() {
                    for (k, v) in custom {
                        options.insert(k.clone(), v.clone());
                    }
                }
            }
        }

        let url = format!("{}/api/generate", endpoint.trim_end_matches('/'));
        let resp = client.post(&url).json(&body).send().await.map_err(|e| {
            PipelineError::Other(format!("Failed to connect to LLM at {}: {}", url, e))
        })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(PipelineError::Other(format!(
                "LLM returned error {}: {}",
                status, text
            )));
        }

        let mut stream = resp.bytes_stream();
        let mut decoder = StreamingDecoder::new();
        let mut accumulated = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(PipelineError::Request)?;
            for json_val in decoder.decode(&chunk) {
                if let Some(response) = json_val.get("response").and_then(|v| v.as_str()) {
                    accumulated.push_str(response);
                    on_token(stage_idx, response);
                }
            }
        }

        // Flush remaining buffer
        if let Some(json_val) = decoder.flush() {
            if let Some(response) = json_val.get("response").and_then(|v| v.as_str()) {
                accumulated.push_str(response);
                on_token(stage_idx, response);
            }
        }

        Ok(accumulated)
    }
}

/// Builder for creating pipelines.
pub struct PipelineBuilder<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Clone,
{
    stages: Vec<Stage>,
    context: PipelineContext,
    cancellation: Option<Arc<AtomicBool>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> PipelineBuilder<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Clone,
{
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            context: PipelineContext::new(),
            cancellation: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a stage to the pipeline.
    pub fn add_stage(mut self, stage: Stage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Set the context for prompt template substitution.
    pub fn with_context(mut self, context: PipelineContext) -> Self {
        self.context = context;
        self
    }

    /// Set a cancellation flag that can be used to abort execution.
    pub fn with_cancellation(mut self, cancel: Arc<AtomicBool>) -> Self {
        self.cancellation = Some(cancel);
        self
    }

    /// Build the pipeline, validating configuration.
    pub fn build(self) -> Result<Pipeline<T>> {
        if self.stages.is_empty() {
            return Err(PipelineError::InvalidConfig(
                "Pipeline must have at least one stage".to_string(),
            ));
        }

        let has_enabled = self.stages.iter().any(|s| s.enabled);
        if !has_enabled {
            return Err(PipelineError::InvalidConfig(
                "Pipeline must have at least one enabled stage".to_string(),
            ));
        }

        Ok(Pipeline {
            stages: self.stages,
            context: self.context,
            cancellation: self.cancellation,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<T> Default for PipelineBuilder<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
    struct TestOutput {
        value: String,
    }

    #[test]
    fn test_pipeline_builder_success() {
        let result = Pipeline::<TestOutput>::builder()
            .add_stage(Stage::new("stage1", "Test: {input}"))
            .add_stage(Stage::new("stage2", "Refine: {input}"))
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_pipeline_fails() {
        let result = Pipeline::<TestOutput>::builder().build();
        assert!(result.is_err());
        match result.unwrap_err() {
            PipelineError::InvalidConfig(msg) => {
                assert!(msg.contains("at least one stage"));
            }
            _ => panic!("Expected InvalidConfig error"),
        }
    }

    #[test]
    fn test_all_disabled_pipeline_fails() {
        let result = Pipeline::<TestOutput>::builder()
            .add_stage(Stage::new("s1", "test").disabled())
            .build();
        assert!(result.is_err());
        match result.unwrap_err() {
            PipelineError::InvalidConfig(msg) => {
                assert!(msg.contains("enabled"));
            }
            _ => panic!("Expected InvalidConfig error"),
        }
    }

    #[test]
    fn test_pipeline_with_context() {
        let context = PipelineContext::new()
            .insert("domain", "science")
            .insert("level", "expert");

        let pipeline = Pipeline::<TestOutput>::builder()
            .add_stage(Stage::new("s1", "{input} in {domain}"))
            .with_context(context)
            .build();
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_pipeline_with_cancellation() {
        let cancel = Arc::new(AtomicBool::new(false));
        let pipeline = Pipeline::<TestOutput>::builder()
            .add_stage(Stage::new("s1", "{input}"))
            .with_cancellation(cancel.clone())
            .build()
            .unwrap();

        // Not cancelled yet
        assert!(pipeline.check_cancelled().is_ok());

        // Set cancelled
        cancel.store(true, Ordering::Relaxed);
        let result = pipeline.check_cancelled();
        assert!(result.is_err());
        match result.unwrap_err() {
            PipelineError::Cancelled => {}
            _ => panic!("Expected Cancelled error"),
        }
    }

    #[test]
    fn test_pipeline_stages_accessor() {
        let pipeline = Pipeline::<TestOutput>::builder()
            .add_stage(Stage::new("a", "p1"))
            .add_stage(Stage::new("b", "p2"))
            .build()
            .unwrap();
        assert_eq!(pipeline.stages().len(), 2);
        assert_eq!(pipeline.stages()[0].name, "a");
        assert_eq!(pipeline.stages()[1].name, "b");
    }

    #[test]
    fn test_build_payloads_skips_disabled() {
        let pipeline = Pipeline::<TestOutput>::builder()
            .add_stage(Stage::new("a", "p1"))
            .add_stage(Stage::new("b", "p2").disabled())
            .add_stage(Stage::new("c", "p3"))
            .build()
            .unwrap();

        let payloads = pipeline.build_payloads(false);
        assert_eq!(payloads.len(), 2);
        assert_eq!(payloads[0].0, 0); // stage index 0
        assert_eq!(payloads[0].1.name(), "a");
        assert_eq!(payloads[1].0, 2); // stage index 2 (b was skipped)
        assert_eq!(payloads[1].1.name(), "c");
    }
}
