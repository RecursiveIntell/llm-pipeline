//! Parse diagnostics and telemetry for output parsing.
//!
//! [`ParseDiagnostics`] records what happened during output parsing — which
//! strategy was used, whether parsing succeeded, how many retries were
//! attempted, and whether repair or auto-completion was involved.

/// Records what happened during output parsing.
///
/// Attached to every [`PayloadOutput`](crate::payload::PayloadOutput) produced
/// by [`LlmCall`](crate::llm_call::LlmCall). Tells the caller which extraction
/// strategy succeeded (or that parsing failed), how many retries were attempted,
/// and whether auto-completion or repair was involved.
///
/// # Example
///
/// ```
/// use llm_pipeline::diagnostics::ParseDiagnostics;
///
/// let diag = ParseDiagnostics::default();
/// assert!(diag.ok()); // No parse_error means success
/// ```
#[derive(Debug, Clone, Default)]
pub struct ParseDiagnostics {
    /// Which parse strategy ultimately produced the Value.
    /// e.g. `"lossy"`, `"json"`, `"string_list"`, `"xml_tag"`, `"custom"`.
    pub strategy: Option<&'static str>,

    /// If parsing failed, the error message. `None` means success.
    pub parse_error: Option<String>,

    /// Number of semantic retry attempts (0 = no retries, initial call succeeded
    /// or retry was not configured).
    pub retry_attempts: u32,

    /// Number of transport retries (429, 5xx) before the request succeeded.
    /// 0 = first attempt succeeded.
    pub transport_retries: u32,

    /// Total time spent in backoff delays (milliseconds).
    pub backoff_total_ms: u64,

    /// Whether JSON repair was applied (trailing commas, single quotes, etc.).
    pub repaired: bool,

    /// Whether auto-completion was used (streaming partial parse).
    pub auto_completed: bool,
}

impl ParseDiagnostics {
    /// Quick check: did parsing succeed?
    pub fn ok(&self) -> bool {
        self.parse_error.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostics_default_is_ok() {
        let d = ParseDiagnostics::default();
        assert!(d.ok());
        assert!(d.strategy.is_none());
        assert_eq!(d.retry_attempts, 0);
        assert_eq!(d.transport_retries, 0);
        assert_eq!(d.backoff_total_ms, 0);
        assert!(!d.repaired);
        assert!(!d.auto_completed);
    }

    #[test]
    fn test_diagnostics_with_error_is_not_ok() {
        let d = ParseDiagnostics {
            parse_error: Some("bad json".to_string()),
            ..Default::default()
        };
        assert!(!d.ok());
    }
}
