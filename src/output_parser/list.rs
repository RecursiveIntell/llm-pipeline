//! String list extraction from LLM responses (generalized `parse_tags`).
//!
//! This module provides [`parse_string_list`] (the successor to `ollama-vision`'s
//! `parse_tags`) and [`parse_string_list_raw`] for general-purpose list extraction.

use std::collections::HashSet;

use crate::output_parser::error::{truncate, ParseError};
use crate::output_parser::extract::{extract_code_block, find_bracketed, preprocess};
use crate::output_parser::repair::try_repair_json;

/// Parse an LLM response into a cleaned list of strings.
///
/// Cleaning: lowercase, trim, deduplicate, filter empties, filter >50 chars.
/// This is the direct successor to `ollama-vision`'s `parse_tags`.
///
/// Strategies (in order):
/// 1. Direct JSON array
/// 2. JSON object with common list keys ("tags", "items", "results", "list")
/// 3. Markdown code block -> JSON array/object
/// 4. Bracket-matched JSON array
/// 5. JSON repair on best candidate
/// 6. Numbered/bulleted list extraction
/// 7. Comma-separated fallback
pub fn parse_string_list(response: &str) -> Result<Vec<String>, ParseError> {
    let items = parse_string_list_inner(response)?;
    Ok(clean_tags(items))
}

/// Parse into a list without tag-specific cleaning.
///
/// No forced lowercase, no length filter, no dedup.
/// For general-purpose list extraction from LLM responses.
pub fn parse_string_list_raw(response: &str) -> Result<Vec<String>, ParseError> {
    let items = parse_string_list_inner(response)?;
    Ok(items
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect())
}

/// Shared inner logic for both list parsers (before cleaning).
fn parse_string_list_inner(response: &str) -> Result<Vec<String>, ParseError> {
    let trimmed = response.trim();

    if trimmed.is_empty() {
        return Err(ParseError::EmptyResponse);
    }

    // Strategy 1: Direct JSON array
    if let Ok(arr) = serde_json::from_str::<Vec<String>>(trimmed) {
        return Ok(arr);
    }

    // Preprocess: strip think tags, trim
    let cleaned = preprocess(trimmed);

    if cleaned.is_empty() {
        return Err(ParseError::EmptyResponse);
    }

    // Strategy 1b: JSON array after preprocessing
    if let Ok(arr) = serde_json::from_str::<Vec<String>>(&cleaned) {
        return Ok(arr);
    }

    // Strategy 2: JSON object with common list keys
    if let Some(tags) = try_extract_list_from_object(&cleaned) {
        return Ok(tags);
    }

    // Strategy 3: Markdown code block extraction
    if let Some(tags) = extract_list_from_code_block(&cleaned) {
        return Ok(tags);
    }

    // Strategy 4: Bracket-matched JSON array
    if let Some(bracket_str) = find_bracketed(&cleaned, '[', ']') {
        if let Ok(arr) = serde_json::from_str::<Vec<String>>(bracket_str) {
            return Ok(arr);
        }
        // Try repair on the bracketed substring
        if let Some(repaired) = try_repair_json(bracket_str) {
            if let Ok(arr) = serde_json::from_str::<Vec<String>>(&repaired) {
                return Ok(arr);
            }
        }
    }

    // Strategy 5: Repair on the full cleaned text
    if let Some(repaired) = try_repair_json(&cleaned) {
        if let Ok(arr) = serde_json::from_str::<Vec<String>>(&repaired) {
            return Ok(arr);
        }
        if let Some(tags) = try_extract_list_from_object(&repaired) {
            return Ok(tags);
        }
    }

    // Strategy 6: Numbered/bulleted list extraction
    if let Some(tags) = extract_from_list(&cleaned) {
        return Ok(tags);
    }

    // Strategy 7: Comma-separated fallback
    let tags: Vec<String> = cleaned
        .split(',')
        .map(|s| s.trim().trim_matches('"').trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if tags.is_empty() {
        return Err(ParseError::Unparseable {
            expected_format: "string list",
            text: truncate(&cleaned, 200),
        });
    }

    Ok(tags)
}

/// Try parsing as a JSON object and extracting an array from common keys.
fn try_extract_list_from_object(text: &str) -> Option<Vec<String>> {
    let val: serde_json::Value = serde_json::from_str(text).ok()?;
    for key in ["tags", "items", "results", "list"] {
        if let Some(arr) = val.get(key).and_then(|v| v.as_array()) {
            let tags: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            if !tags.is_empty() {
                return Some(tags);
            }
        }
    }
    None
}

/// Extract a list from a markdown code block.
fn extract_list_from_code_block(text: &str) -> Option<Vec<String>> {
    // Try ```json blocks first, then any block
    if let Some((lang, content)) = extract_code_block(text) {
        // Try direct array parse
        if let Ok(arr) = serde_json::from_str::<Vec<String>>(content) {
            return Some(arr);
        }
        // Try object with list keys
        if let Some(tags) = try_extract_list_from_object(content) {
            return Some(tags);
        }
        // If it was a json block and still failed, try repair
        if lang == Some("json") {
            if let Some(repaired) = try_repair_json(content) {
                if let Ok(arr) = serde_json::from_str::<Vec<String>>(&repaired) {
                    return Some(arr);
                }
            }
        }
    }
    None
}

/// Extract tags from numbered or bulleted lists.
fn extract_from_list(text: &str) -> Option<Vec<String>> {
    let lines: Vec<&str> = text.lines().collect();
    let list_items: Vec<String> = lines
        .iter()
        .filter_map(|line| {
            let trimmed = line.trim();
            // Numbered: "1. tag", "2) tag"
            if let Some(rest) = trimmed
                .strip_prefix(|c: char| c.is_ascii_digit())
                .and_then(|s| {
                    // Handle multi-digit numbers
                    let s = s.trim_start_matches(|c: char| c.is_ascii_digit());
                    s.strip_prefix('.').or_else(|| s.strip_prefix(')'))
                })
            {
                let tag = rest.trim().trim_matches('"').trim();
                if !tag.is_empty() {
                    return Some(tag.to_string());
                }
            }
            // Bulleted: "- tag", "* tag", "• tag"
            for prefix in ["-", "*", "\u{2022}"] {
                if let Some(rest) = trimmed.strip_prefix(prefix) {
                    let tag = rest.trim().trim_matches('"').trim();
                    if !tag.is_empty() {
                        return Some(tag.to_string());
                    }
                }
            }
            None
        })
        .collect();

    if list_items.len() >= 2 {
        Some(list_items)
    } else {
        None
    }
}

/// Clean a list of tags: lowercase, trim, deduplicate, filter empties and long items.
fn clean_tags(tags: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    tags.into_iter()
        .map(|t| t.trim().to_lowercase())
        .filter(|t| !t.is_empty() && t.len() < 50 && seen.insert(t.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ══════════════════════════════════════════════════════════
    // Ported from ollama-vision/src/parser.rs (24 tests)
    // ══════════════════════════════════════════════════════════

    // ── Strategy 1: Direct JSON array ──

    #[test]
    fn parse_json_array() {
        let input = r#"["portrait", "fantasy", "dark lighting"]"#;
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["portrait", "fantasy", "dark lighting"]);
    }

    // ── Strategy 2: Think tags + JSON ──

    #[test]
    fn parse_with_think_blocks() {
        let input = r#"<think>
Let me analyze this image. I see a portrait with dark lighting...
</think>

["portrait", "dark lighting", "woman"]"#;
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["portrait", "dark lighting", "woman"]);
    }

    #[test]
    fn parse_with_incomplete_think_block() {
        let input = "<think>\nStill thinking...\n[\"portrait\", \"fantasy\"]";
        let result = parse_string_list(input);
        assert!(result.is_err());
    }

    #[test]
    fn strip_think_tags_complete() {
        let input = "<think>reasoning</think>result";
        assert_eq!(crate::output_parser::strip_think_tags(input), "result");
    }

    #[test]
    fn strip_think_tags_incomplete() {
        let input = "<think>reasoning without close";
        assert_eq!(crate::output_parser::strip_think_tags(input), "");
    }

    #[test]
    fn strip_think_tags_multiple() {
        let input = "<think>first</think>middle<think>second</think>end";
        assert_eq!(crate::output_parser::strip_think_tags(input), "middleend");
    }

    // ── Strategy 3: JSON object with "tags" key ──

    #[test]
    fn parse_object_with_tags_key() {
        let input = r#"{"tags": ["portrait", "dark", "moody"]}"#;
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["portrait", "dark", "moody"]);
    }

    #[test]
    fn parse_think_then_object() {
        let input = r#"<think>Looking at this...</think>{"tags": ["cat", "cute", "indoor"]}"#;
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["cat", "cute", "indoor"]);
    }

    // ── Strategy 4: Markdown code blocks ──

    #[test]
    fn parse_markdown_code_block() {
        let input =
            "Here are the tags:\n\n```json\n[\"portrait\", \"fantasy\", \"oil painting\"]\n```";
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["portrait", "fantasy", "oil painting"]);
    }

    #[test]
    fn parse_think_then_code_block() {
        let input = "<think>\nAnalyzing...\n</think>\n\n```json\n[\"landscape\", \"sunset\"]\n```";
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["landscape", "sunset"]);
    }

    #[test]
    fn parse_code_block_with_object() {
        let input = "```json\n{\"tags\": [\"a\", \"b\"]}\n```";
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["a", "b"]);
    }

    // ── Strategy 5: Bracket matching ──

    #[test]
    fn parse_with_surrounding_text() {
        let input = r#"Here are the tags: ["cat", "cute", "indoor"]"#;
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["cat", "cute", "indoor"]);
    }

    #[test]
    fn parse_mixed_text_and_json() {
        let input = "I found these:\n[\"a\", \"b\"]\nHope that helps!";
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["a", "b"]);
    }

    // ── Strategy 6: List extraction ──

    #[test]
    fn parse_numbered_list() {
        let input = "1. portrait\n2. fantasy\n3. dark lighting";
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["portrait", "fantasy", "dark lighting"]);
    }

    #[test]
    fn parse_bulleted_list() {
        let input = "- portrait\n- fantasy\n- dark lighting";
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["portrait", "fantasy", "dark lighting"]);
    }

    #[test]
    fn parse_star_bulleted_list() {
        let input = "* cat\n* cute\n* fluffy";
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["cat", "cute", "fluffy"]);
    }

    // ── Strategy 7: Comma-separated fallback ──

    #[test]
    fn parse_comma_separated() {
        let input = "portrait, fantasy, dark lighting";
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["portrait", "fantasy", "dark lighting"]);
    }

    // ── Edge cases ──

    #[test]
    fn parse_empty_fails() {
        assert!(parse_string_list("").is_err());
        assert!(parse_string_list("   ").is_err());
    }

    #[test]
    fn parse_cleans_whitespace_and_case() {
        let input = r#"["  Portrait  ", " FANTASY ", "Dark Lighting"]"#;
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["portrait", "fantasy", "dark lighting"]);
    }

    #[test]
    fn parse_deduplicates() {
        let input = r#"["cat", "Cat", "CAT", "dog"]"#;
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["cat", "dog"]);
    }

    #[test]
    fn parse_filters_long_tags() {
        let input = format!(r#"["good", "{}"]"#, "x".repeat(60));
        let tags = parse_string_list(&input).unwrap();
        assert_eq!(tags, vec!["good"]);
    }

    #[test]
    fn clean_tags_filters_empty() {
        let tags = vec!["good".to_string(), "".to_string(), "  ".to_string()];
        let cleaned = clean_tags(tags);
        assert_eq!(cleaned, vec!["good"]);
    }

    // ══════════════════════════════════════════════════════════
    // New tests for generalized functionality
    // ══════════════════════════════════════════════════════════

    #[test]
    fn json_object_with_items_key() {
        let input = r#"{"items": ["a", "b"]}"#;
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["a", "b"]);
    }

    #[test]
    fn json_object_with_results_key() {
        let input = r#"{"results": ["a", "b"]}"#;
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["a", "b"]);
    }

    #[test]
    fn json_object_with_list_key() {
        let input = r#"{"list": ["a", "b"]}"#;
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["a", "b"]);
    }

    #[test]
    fn repaired_json_list() {
        let input = "['tag1', 'tag2']";
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["tag1", "tag2"]);
    }

    #[test]
    fn raw_preserves_case() {
        let input = r#"["Alpha", "Beta"]"#;
        let tags = parse_string_list_raw(input).unwrap();
        assert_eq!(tags, vec!["Alpha", "Beta"]);
    }

    #[test]
    fn raw_preserves_length() {
        let long_item = "x".repeat(60);
        let input = format!(r#"["short", "{}"]"#, long_item);
        let tags = parse_string_list_raw(&input).unwrap();
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[1], long_item);
    }

    #[test]
    fn raw_comma_separated() {
        let input = "Alpha, Beta, Gamma";
        let tags = parse_string_list_raw(input).unwrap();
        assert_eq!(tags, vec!["Alpha", "Beta", "Gamma"]);
    }

    #[test]
    fn thinking_tag_variant() {
        let input = r#"<thinking>analyzing...</thinking>["a", "b"]"#;
        let tags = parse_string_list(input).unwrap();
        assert_eq!(tags, vec!["a", "b"]);
    }
}
