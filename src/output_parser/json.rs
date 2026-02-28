//! Typed JSON extraction from LLM responses.
//!
//! Provides [`parse_json`] for extracting typed structs and [`parse_json_value`]
//! for untyped JSON extraction, using a multi-strategy pipeline that handles
//! think blocks, markdown fences, bracket matching, and JSON repair.

use serde::de::DeserializeOwned;

use crate::output_parser::error::{truncate, ParseError};
use crate::output_parser::extract::{
    extract_code_block, extract_code_block_for, find_bracketed, preprocess,
};
use crate::output_parser::repair::try_repair_json;
use crate::output_parser::streaming::auto_complete_json;

/// Parse an LLM response into a typed struct.
///
/// Strategies (in order):
/// 1. Direct deserialize on preprocessed text
/// 2. Extract from markdown code block (`` ```json ``)
/// 3. Extract from any code block
/// 4. Bracket-match a JSON object (`{...}`)
/// 5. Bracket-match a JSON array (`[...]`)
/// 6. Repair malformed JSON then retry strategies 1-5
///
/// # Examples
///
/// ```
/// use serde::Deserialize;
/// use llm_pipeline::output_parser::parse_json;
///
/// #[derive(Deserialize, Debug, PartialEq)]
/// struct Analysis {
///     sentiment: String,
///     confidence: f64,
/// }
///
/// let response = r#"<think>analyzing...</think>{"sentiment": "positive", "confidence": 0.92}"#;
/// let result: Analysis = parse_json(response).unwrap();
/// assert_eq!(result.sentiment, "positive");
/// ```
pub fn parse_json<T: DeserializeOwned>(response: &str) -> Result<T, ParseError> {
    let (candidate, cleaned) = extract_json_candidate(response)?;

    // Try deserializing the candidate
    let deser_err = match serde_json::from_str::<T>(&candidate) {
        Ok(val) => return Ok(val),
        Err(e) => e.to_string(),
    };

    // Try repair on the candidate
    if let Some(repaired) = try_repair_json(&candidate) {
        if let Ok(val) = serde_json::from_str::<T>(&repaired) {
            return Ok(val);
        }
    }

    // Try repair on the full cleaned text if different from candidate
    if candidate != cleaned {
        if let Some(repaired) = try_repair_json(&cleaned) {
            if let Ok(val) = serde_json::from_str::<T>(&repaired) {
                return Ok(val);
            }
        }
    }

    // Try auto-completing truncated JSON as final strategy
    if let Some(completed) = auto_complete_json(&candidate) {
        if let Ok(val) = serde_json::from_str::<T>(&completed) {
            return Ok(val);
        }
    }

    // All strategies exhausted
    Err(ParseError::DeserializationFailed {
        reason: deser_err,
        raw_json: truncate(&candidate, 200),
    })
}

/// Parse into a `serde_json::Value` when you don't know the schema.
///
/// Uses the same strategy pipeline as [`parse_json`].
pub fn parse_json_value(response: &str) -> Result<serde_json::Value, ParseError> {
    parse_json(response)
}

/// Try all extraction strategies and return the best JSON candidate string.
/// Returns `(best_candidate, cleaned_text)`.
fn extract_json_candidate(response: &str) -> Result<(String, String), ParseError> {
    let trimmed = response.trim();

    if trimmed.is_empty() {
        return Err(ParseError::EmptyResponse);
    }

    let cleaned = preprocess(trimmed);

    if cleaned.is_empty() {
        return Err(ParseError::EmptyResponse);
    }

    // Strategy 1: Direct parse on cleaned text
    if serde_json::from_str::<serde_json::Value>(&cleaned).is_ok() {
        return Ok((cleaned.clone(), cleaned));
    }

    // Strategy 2: Extract from ```json code block
    if let Some(content) = extract_code_block_for(&cleaned, "json") {
        if serde_json::from_str::<serde_json::Value>(content).is_ok() {
            return Ok((content.to_string(), cleaned));
        }
        // Even if not valid yet, this is a good candidate for repair
        return Ok((content.to_string(), cleaned));
    }

    // Strategy 3: Extract from any code block
    if let Some((_lang, content)) = extract_code_block(&cleaned) {
        // Check if it looks like JSON (starts with { or [)
        let trimmed_content = content.trim();
        if trimmed_content.starts_with('{') || trimmed_content.starts_with('[') {
            if serde_json::from_str::<serde_json::Value>(trimmed_content).is_ok() {
                return Ok((trimmed_content.to_string(), cleaned));
            }
            return Ok((trimmed_content.to_string(), cleaned));
        }
    }

    // Strategy 4: Bracket-match a JSON object
    if let Some(bracket_str) = find_bracketed(&cleaned, '{', '}') {
        if serde_json::from_str::<serde_json::Value>(bracket_str).is_ok() {
            return Ok((bracket_str.to_string(), cleaned));
        }
        return Ok((bracket_str.to_string(), cleaned));
    }

    // Strategy 5: Bracket-match a JSON array
    if let Some(bracket_str) = find_bracketed(&cleaned, '[', ']') {
        if serde_json::from_str::<serde_json::Value>(bracket_str).is_ok() {
            return Ok((bracket_str.to_string(), cleaned));
        }
        return Ok((bracket_str.to_string(), cleaned));
    }

    // No candidate found — return cleaned text as the candidate for repair
    Ok((cleaned.clone(), cleaned))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct Kv {
        key: String,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Sentiment {
        sentiment: String,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Outer {
        outer: Inner,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Inner {
        inner: Vec<i32>,
    }

    #[test]
    fn direct_json_object() {
        let input = r#"{"key": "value"}"#;
        let result: Kv = parse_json(input).unwrap();
        assert_eq!(result.key, "value");
    }

    #[test]
    fn direct_json_array() {
        let input = "[1, 2, 3]";
        let result: Vec<i32> = parse_json(input).unwrap();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn think_then_json() {
        let input = r#"<think>analyzing</think>{"key": "value"}"#;
        let result: Kv = parse_json(input).unwrap();
        assert_eq!(result.key, "value");
    }

    #[test]
    fn code_block_json() {
        let input = "Here's the data:\n```json\n{\"key\": \"value\"}\n```";
        let result: Kv = parse_json(input).unwrap();
        assert_eq!(result.key, "value");
    }

    #[test]
    fn bare_code_block() {
        let input = "```\n{\"key\": \"value\"}\n```";
        let result: Kv = parse_json(input).unwrap();
        assert_eq!(result.key, "value");
    }

    #[test]
    fn json_in_prose() {
        let input = r#"The analysis is {"sentiment": "positive"} as shown."#;
        let result: Sentiment = parse_json(input).unwrap();
        assert_eq!(result.sentiment, "positive");
    }

    #[test]
    fn nested_json() {
        let input = r#"{"outer": {"inner": [1,2,3]}}"#;
        let result: Outer = parse_json(input).unwrap();
        assert_eq!(result.outer.inner, vec![1, 2, 3]);
    }

    #[test]
    fn repaired_trailing_comma() {
        let input = r#"{"key": "value",}"#;
        let result: Kv = parse_json(input).unwrap();
        assert_eq!(result.key, "value");
    }

    #[test]
    fn repaired_single_quotes() {
        let input = "{'key': 'value'}";
        let result: Kv = parse_json(input).unwrap();
        assert_eq!(result.key, "value");
    }

    #[test]
    fn think_and_code_block() {
        let input = "<think>hmm</think>\n```json\n{\"key\": \"value\"}\n```";
        let result: Kv = parse_json(input).unwrap();
        assert_eq!(result.key, "value");
    }

    #[test]
    fn json_with_surrounding_text() {
        let input = "Sure! Here's your result: {\"key\": \"value\"}\nHope that helps!";
        let result: Kv = parse_json(input).unwrap();
        assert_eq!(result.key, "value");
    }

    #[test]
    fn parse_json_value_works() {
        let input = r#"{"a": 1, "b": "two"}"#;
        let val = parse_json_value(input).unwrap();
        assert_eq!(val["a"], 1);
        assert_eq!(val["b"], "two");
    }

    #[test]
    fn empty_response_fails() {
        let result: Result<Kv, _> = parse_json("");
        assert!(result.is_err());
    }
}
