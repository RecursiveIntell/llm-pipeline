use std::time::Duration;
use thiserror::Error;

/// Errors produced by the pipeline and its components.
#[derive(Error, Debug)]
pub enum PipelineError {
    /// Low-level HTTP transport failure (connection refused, timeout, etc.).
    #[error("HTTP request failed: {0}")]
    Request(#[from] reqwest::Error),

    /// JSON parsing failed at the serde level.
    #[error("JSON parsing failed: {0}")]
    Json(#[from] serde_json::Error),

    /// A pipeline stage failed with a descriptive message.
    #[error("Stage '{stage}' failed: {message}")]
    StageFailed { stage: String, message: String },

    /// The pipeline or payload was cancelled via the cancellation flag.
    #[error("Pipeline was cancelled")]
    Cancelled,

    /// Invalid configuration detected at build time.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// HTTP error with status code, response body, and optional Retry-After hint.
    ///
    /// Returned by [`Backend`](crate::backend::Backend) implementations when
    /// the provider returns a non-success status code. The `retry_after` field
    /// is populated from the `Retry-After` response header when present.
    #[error("HTTP {status}: {body}")]
    HttpError {
        /// HTTP status code (e.g. 429, 500, 503).
        status: u16,
        /// Response body text.
        body: String,
        /// Parsed `Retry-After` header value, if present.
        retry_after: Option<Duration>,
    },

    /// Catch-all for other errors.
    #[error("{0}")]
    Other(String),
}

impl From<anyhow::Error> for PipelineError {
    fn from(err: anyhow::Error) -> Self {
        PipelineError::Other(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, PipelineError>;
