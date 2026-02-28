//! Parsing utilities for LLM responses.
//!
//! Defensive extraction of JSON, thinking blocks, and typed output from
//! potentially messy LLM text. These utilities are the core of what makes
//! LLM output reliable enough for structured workflows.

use crate::error::Result;
use crate::PipelineError;
use serde::de::DeserializeOwned;
use serde_json::Value;

/// Extract `<think>...</think>` blocks from a response (DeepSeek R1 style).
///
/// Returns `(thinking_content, cleaned_text)` where `cleaned_text` has the
/// thinking block removed and is trimmed.
pub fn extract_thinking(text: &str) -> (Option<String>, String) {
    let think_start = "<think>";
    let think_end = "</think>";

    if let Some(start_idx) = text.find(think_start) {
        if let Some(end_idx) = text.find(think_end) {
            let thinking = text[start_idx + think_start.len()..end_idx]
                .trim()
                .to_string();
            let mut cleaned = String::new();
            cleaned.push_str(&text[..start_idx]);
            cleaned.push_str(&text[end_idx + think_end.len()..]);
            let cleaned = cleaned.trim().to_string();
            let thinking = if thinking.is_empty() {
                None
            } else {
                Some(thinking)
            };
            return (thinking, cleaned);
        }
    }

    (None, text.to_string())
}

/// Extract JSON content from markdown fenced code blocks.
///
/// Recognizes `` ```json ``, `` ```JSON ``, and plain `` ``` `` fences.
pub fn extract_json_block(text: &str) -> Option<String> {
    let markers = ["```json", "```JSON", "```"];
    for marker in markers {
        if let Some(start) = text.find(marker) {
            let content_start = start + marker.len();
            if let Some(end) = text[content_start..].find("```") {
                return Some(text[content_start..content_start + end].trim().to_string());
            }
        }
    }
    None
}

/// Try to locate and extract a JSON object or array from text that may
/// contain surrounding prose.
///
/// Tries, in order:
/// 1. Markdown code block extraction
/// 2. First `{` or `[` with matching closer
pub fn extract_json_candidate(text: &str) -> Option<String> {
    let trimmed = text.trim();

    // Try markdown block first
    if let Some(block) = extract_json_block(trimmed) {
        return Some(block);
    }

    // Find first { or [
    if let Some(idx) = trimmed.find('{').or_else(|| trimmed.find('[')) {
        let candidate = &trimmed[idx..];
        // Try parsing the whole remainder
        if serde_json::from_str::<Value>(candidate).is_ok() {
            return Some(candidate.to_string());
        }
        // Try finding matching closing brace/bracket
        let open = candidate.as_bytes()[0];
        let close = if open == b'{' { b'}' } else { b']' };
        if let Some(end) = candidate.rfind(close as char) {
            let substr = &candidate[..=end];
            if serde_json::from_str::<Value>(substr).is_ok() {
                return Some(substr.to_string());
            }
        }
    }

    None
}

/// Parse text into a `serde_json::Value` losslessly.
///
/// Tries defensive JSON extraction. If no valid JSON is found, wraps
/// the text as `Value::String`. This function never fails.
pub fn parse_value_lossy(text: &str) -> Value {
    let trimmed = text.trim();

    // Direct parse
    if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
        return val;
    }

    // Extract from blocks or embedded JSON
    if let Some(candidate) = extract_json_candidate(trimmed) {
        if let Ok(val) = serde_json::from_str::<Value>(&candidate) {
            return val;
        }
    }

    // Fall back to wrapping as string
    Value::String(trimmed.to_string())
}

/// Parse text into a `serde_json::Value`, requiring valid JSON.
///
/// Tries defensive extraction but returns an error if no valid JSON is found.
pub fn parse_value_defensively(text: &str) -> Result<Value> {
    let trimmed = text.trim();

    if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
        return Ok(val);
    }

    if let Some(candidate) = extract_json_candidate(trimmed) {
        if let Ok(val) = serde_json::from_str::<Value>(&candidate) {
            return Ok(val);
        }
    }

    Err(PipelineError::Other(format!(
        "No valid JSON found in LLM output. Raw text (truncated): {}",
        &trimmed[..trimmed.len().min(200)]
    )))
}

/// Parse text into a typed `T` with defensive JSON extraction.
///
/// Tries direct parse, markdown block extraction, and embedded JSON detection.
/// This is the typed equivalent of [`parse_value_defensively`].
pub fn parse_as<T: DeserializeOwned>(text: &str) -> Result<T> {
    let trimmed = text.trim();

    // Try direct parse
    if let Ok(val) = serde_json::from_str::<T>(trimmed) {
        return Ok(val);
    }

    // Try extracting JSON from markdown code blocks
    if let Some(json_str) = extract_json_block(trimmed) {
        if let Ok(val) = serde_json::from_str::<T>(&json_str) {
            return Ok(val);
        }
    }

    // Try finding first { or [ and parsing from there
    if let Some(idx) = trimmed.find('{').or_else(|| trimmed.find('[')) {
        let candidate = &trimmed[idx..];
        if let Ok(val) = serde_json::from_str::<T>(candidate) {
            return Ok(val);
        }
        let open = candidate.as_bytes()[0];
        let close = if open == b'{' { b'}' } else { b']' };
        if let Some(end) = candidate.rfind(close as char) {
            let substr = &candidate[..=end];
            if let Ok(val) = serde_json::from_str::<T>(substr) {
                return Ok(val);
            }
        }
    }

    Err(PipelineError::Other(format!(
        "Failed to parse LLM output as expected type. Raw text (truncated): {}",
        &trimmed[..trimmed.len().min(200)]
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_thinking_present() {
        let text = "Before <think>my reasoning here</think> after";
        let (thinking, cleaned) = extract_thinking(text);
        assert_eq!(thinking, Some("my reasoning here".to_string()));
        assert_eq!(cleaned, "Before  after");
    }

    #[test]
    fn test_extract_thinking_absent() {
        let text = "no thinking tags here";
        let (thinking, cleaned) = extract_thinking(text);
        assert!(thinking.is_none());
        assert_eq!(cleaned, "no thinking tags here");
    }

    #[test]
    fn test_extract_thinking_empty() {
        let text = "<think>  </think>actual content";
        let (thinking, cleaned) = extract_thinking(text);
        assert!(thinking.is_none());
        assert_eq!(cleaned, "actual content");
    }

    #[test]
    fn test_extract_json_block() {
        let text = "text\n```json\n{\"a\":1}\n```\nmore";
        assert_eq!(extract_json_block(text), Some("{\"a\":1}".to_string()));
    }

    #[test]
    fn test_extract_json_block_none() {
        assert_eq!(extract_json_block("no code block"), None);
    }

    #[test]
    fn test_extract_json_candidate_from_block() {
        let text = "```json\n{\"x\":1}\n```";
        assert_eq!(extract_json_candidate(text), Some("{\"x\":1}".to_string()));
    }

    #[test]
    fn test_extract_json_candidate_embedded() {
        let text = "Here is the result: {\"name\": \"test\"} done.";
        let candidate = extract_json_candidate(text);
        assert!(candidate.is_some());
        let val: Value = serde_json::from_str(&candidate.unwrap()).unwrap();
        assert_eq!(val["name"], "test");
    }

    #[test]
    fn test_parse_value_lossy_json() {
        let val = parse_value_lossy(r#"{"key": "value"}"#);
        assert_eq!(val["key"], "value");
    }

    #[test]
    fn test_parse_value_lossy_string() {
        let val = parse_value_lossy("just plain text");
        assert_eq!(val, Value::String("just plain text".to_string()));
    }

    #[test]
    fn test_parse_value_defensively_ok() {
        let val = parse_value_defensively(r#"{"a": 1}"#).unwrap();
        assert_eq!(val["a"], 1);
    }

    #[test]
    fn test_parse_value_defensively_err() {
        let result = parse_value_defensively("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_as_direct() {
        #[derive(Debug, serde::Deserialize, PartialEq)]
        struct T {
            value: String,
        }
        let result: T = parse_as(r#"{"value": "hello"}"#).unwrap();
        assert_eq!(result.value, "hello");
    }

    #[test]
    fn test_parse_as_markdown_block() {
        #[derive(Debug, serde::Deserialize, PartialEq)]
        struct T {
            x: i32,
        }
        let text = "Here:\n```json\n{\"x\": 42}\n```\nDone.";
        let result: T = parse_as(text).unwrap();
        assert_eq!(result.x, 42);
    }

    #[test]
    fn test_parse_as_embedded() {
        #[derive(Debug, serde::Deserialize, PartialEq)]
        struct T {
            name: String,
        }
        let text = "Sure! {\"name\": \"test\"} hope that helps.";
        let result: T = parse_as(text).unwrap();
        assert_eq!(result.name, "test");
    }

    #[test]
    fn test_parse_as_failure() {
        #[derive(Debug, serde::Deserialize)]
        struct T {
            _x: i32,
        }
        let result = parse_as::<T>("not json at all");
        assert!(result.is_err());
    }
}
