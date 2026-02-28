//! Clean text extraction from LLM responses.
//!
//! Provides [`parse_text`] for extracting clean prose from LLM output,
//! stripping think blocks and common boilerplate prefixes.

use crate::output_parser::error::ParseError;
use crate::output_parser::extract::preprocess;

/// Common boilerplate prefixes that LLMs add to responses.
const SIMPLE_PREFIXES: &[&str] = &[
    "Sure! ",
    "Sure, ",
    "Sure.\n",
    "Of course! ",
    "Of course, ",
    "Of course.\n",
    "Certainly! ",
    "Certainly, ",
    "Certainly.\n",
    "Absolutely! ",
    "Absolutely, ",
];

/// Prefixes that consume up to the next newline or colon.
const LINE_PREFIXES: &[&str] = &["Here's ", "Here is "];

/// Clean an LLM response for use as plain text.
///
/// Processing:
/// 1. Strip `<think>` blocks
/// 2. Trim whitespace
/// 3. Strip common LLM boilerplate prefixes:
///    "Sure!", "Here's...", "Of course!", "Certainly!", etc.
///
/// Returns the cleaned text or `EmptyResponse` if nothing remains.
///
/// # Examples
///
/// ```
/// use llm_pipeline::output_parser::parse_text;
///
/// let result = parse_text("Sure! Paris is the capital.").unwrap();
/// assert_eq!(result, "Paris is the capital.");
/// ```
pub fn parse_text(response: &str) -> Result<String, ParseError> {
    let cleaned = preprocess(response);

    if cleaned.is_empty() {
        return Err(ParseError::EmptyResponse);
    }

    let mut text = cleaned.as_str();

    // Try stripping simple prefixes
    for prefix in SIMPLE_PREFIXES {
        if let Some(rest) = text.strip_prefix(prefix) {
            text = rest;
            break;
        }
    }

    // Try stripping line prefixes (consume up to next newline or colon)
    if text == cleaned.as_str() {
        // Only if no simple prefix was stripped
        for prefix in LINE_PREFIXES {
            if let Some(rest) = text.strip_prefix(prefix) {
                // Find the next newline or colon and skip past it
                if let Some(pos) = rest.find('\n') {
                    text = rest[pos + 1..].trim_start();
                    break;
                } else if let Some(pos) = rest.find(':') {
                    text = rest[pos + 1..].trim_start();
                    break;
                }
            }
        }
    }

    let result = text.trim().to_string();

    if result.is_empty() {
        return Err(ParseError::EmptyResponse);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_text() {
        let result = parse_text("Paris is the capital.").unwrap();
        assert_eq!(result, "Paris is the capital.");
    }

    #[test]
    fn with_think() {
        let result = parse_text("<think>reasoning</think>Paris.").unwrap();
        assert_eq!(result, "Paris.");
    }

    #[test]
    fn sure_prefix() {
        let result = parse_text("Sure! Paris is the capital.").unwrap();
        assert_eq!(result, "Paris is the capital.");
    }

    #[test]
    fn heres_prefix() {
        let result = parse_text("Here's the answer:\nParis.").unwrap();
        assert_eq!(result, "Paris.");
    }

    #[test]
    fn empty_after_strip() {
        let result = parse_text("<think>just thinking</think>");
        assert!(result.is_err());
    }

    #[test]
    fn already_clean() {
        let result = parse_text("No prefix here.").unwrap();
        assert_eq!(result, "No prefix here.");
    }
}
