//! Error types for LLM output parsers.

/// Errors returned by output parsers.
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    /// The LLM response was empty or whitespace-only.
    #[error("empty LLM response")]
    EmptyResponse,

    /// No parsing strategy could extract the expected format.
    #[error("could not parse {expected_format} from LLM response: {text}")]
    Unparseable {
        /// The format the parser was trying to extract.
        expected_format: &'static str,
        /// A truncated copy of the cleaned LLM text (max 200 chars).
        text: String,
    },

    /// JSON was extracted but failed to deserialize into the target type.
    #[error("JSON deserialization failed: {reason}")]
    DeserializationFailed {
        /// The serde error message.
        reason: String,
        /// The raw JSON string that failed deserialization.
        raw_json: String,
    },

    /// No valid choice from the provided options was found.
    #[error("no valid choice found in response (valid: {valid:?})")]
    NoMatchingChoice {
        /// The list of choices that were searched for.
        valid: Vec<String>,
    },

    /// No number found, or number was outside the expected range.
    #[error("no valid number found in response")]
    NoNumber,
}

/// Truncate a string to at most `max_len` characters, appending "..." if truncated.
#[allow(dead_code)]
pub(crate) fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
