use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Receipt types for pipeline execution auditing
// ---------------------------------------------------------------------------

/// Outcome of a pipeline execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionOutcome {
    Success,
    StageFailed(String),
    Cancelled,
    InternalError(String),
}

/// Retry cause classification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RetryCause {
    ParseError(String),
    ValidationError(String),
    TransportError(String),
    RateLimited,
    Other(String),
}

/// Retry decision classification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RetryDecision {
    Retrying,
    GivingUp,
    FallingBack(String),
}

/// Budget debit record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetDebitV1 {
    pub budget_id: String,
    pub debit: f64,
    pub remaining: f64,
}

/// Provider call receipt — one per LLM call made during pipeline execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCallReceiptV1 {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub integrity_tag: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_receipt_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub traceparent: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tracestate: Option<String>,
    pub receipt_id: String,
    pub provider: String,
    pub model_route: String,
    pub request_digest: String,
    pub response_digest: String,
    pub latency_ms: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
}

/// Retry decision receipt — one per retry attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryDecisionReceiptV1 {
    pub receipt_id: String,
    pub attempt_number: u32,
    pub max_attempts: u32,
    pub cause: RetryCause,
    pub decision: RetryDecision,
    pub budget_impact: BudgetDebitV1,
}

/// Complete pipeline execution receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineExecutionReceiptV1 {
    #[serde(default = "default_receipt_version")]
    pub receipt_version: String,
    #[serde(default = "default_crate_version")]
    pub crate_version: String,
    #[serde(default)]
    pub integrity_tag: Option<String>,
    #[serde(default)]
    pub previous_receipt_digest: Option<String>,
    #[serde(default)]
    pub traceparent: Option<String>,
    #[serde(default)]
    pub tracestate: Option<String>,
    #[serde(default)]
    pub chain_valid: bool,
    pub receipt_id: String,
    pub pipeline_id: String,
    pub provider_calls: Vec<ProviderCallReceiptV1>,
    pub retry_decisions: Vec<RetryDecisionReceiptV1>,
    pub budget_debits: Vec<BudgetDebitV1>,
    pub response_digest: String,
    pub outcome: ExecutionOutcome,
    pub recorded_time: chrono::DateTime<chrono::Utc>,
}

fn default_receipt_version() -> String {
    "1".to_string()
}
fn default_crate_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Input to a pipeline execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineInput {
    /// The main input text/idea.
    pub idea: String,

    /// Number of variations to generate (for ideation stages).
    pub num_concepts: u32,
}

impl PipelineInput {
    pub fn new(idea: impl Into<String>) -> Self {
        Self {
            idea: idea.into(),
            num_concepts: 3,
        }
    }

    pub fn with_concepts(mut self, num: u32) -> Self {
        self.num_concepts = num;
        self
    }
}

/// Output from a single stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageOutput<T> {
    /// The structured output parsed from the LLM response.
    pub output: T,

    /// Thinking process (if extended thinking was enabled).
    pub thinking: Option<String>,

    /// Raw response text from the LLM.
    pub raw_response: String,
}

/// Complete pipeline execution result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult<T> {
    /// Final output from the last executed stage.
    pub final_output: T,

    /// Results from each executed stage in order.
    pub stage_results: Vec<StageOutput<T>>,

    /// Which stages were enabled during execution (indexed by stage position).
    pub stages_enabled: Vec<bool>,
}

/// Progress update emitted during pipeline execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineProgress {
    /// Current stage index (0-based).
    pub stage_index: usize,

    /// Total number of stages in the pipeline.
    pub total_stages: usize,

    /// Name of the current stage.
    pub stage_name: String,

    /// Current step within the stage (optional).
    pub current_step: Option<u32>,

    /// Total steps in the stage (optional).
    pub total_steps: Option<u32>,
}

/// Context that can be injected into prompt templates via `{key}` placeholders.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineContext {
    pub data: HashMap<String, String>,
}

impl PipelineContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.data.get(key).map(|s| s.as_str())
    }
}
