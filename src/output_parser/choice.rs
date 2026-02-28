//! Choice extraction from LLM responses.
//!
//! Provides [`parse_choice`] for extracting a single choice from a set of
//! valid options, handling common LLM formatting patterns like bold, quotes,
//! and prose wrapping.

use crate::output_parser::error::ParseError;
use crate::output_parser::extract::preprocess;

/// Extract a single choice from a set of valid options.
///
/// Handles common LLM response patterns:
/// - Direct match: `"positive"`
/// - Bold: `"**positive**"`
/// - Quoted: `"'positive'"` or `"\"positive\""`
/// - In prose: `"I would classify this as positive because..."`
/// - Parenthesized: `"(positive)"`
///
/// Matching is case-insensitive. If multiple valid choices appear,
/// returns the first one found in the text.
///
/// # Examples
///
/// ```
/// use llm_pipeline::output_parser::parse_choice;
///
/// let result = parse_choice("I'd classify this as positive", &["positive", "negative"]).unwrap();
/// assert_eq!(result, "positive");
/// ```
pub fn parse_choice<'a>(response: &str, valid_choices: &[&'a str]) -> Result<&'a str, ParseError> {
    let cleaned = preprocess(response);

    if cleaned.is_empty() {
        return Err(ParseError::EmptyResponse);
    }

    let lower = cleaned.to_lowercase();

    // Strip common wrappers for exact matching
    let stripped = lower
        .trim_matches(|c: char| c == '.' || c == '!' || c == ',' || c.is_whitespace())
        .trim_start_matches("**")
        .trim_end_matches("**")
        .trim_matches('"')
        .trim_matches('\'')
        .trim_matches('(')
        .trim_matches(')')
        .trim();

    // Strategy 1: Exact match on stripped text
    for &choice in valid_choices {
        if stripped.eq_ignore_ascii_case(choice) {
            return Ok(choice);
        }
    }

    // Strategy 2: Stripped text starts with a choice
    for &choice in valid_choices {
        let choice_lower = choice.to_lowercase();
        if stripped.starts_with(&choice_lower) {
            // Check word boundary after the choice
            let after = stripped.len().min(choice_lower.len());
            if after == stripped.len() || !stripped.as_bytes()[after].is_ascii_alphanumeric() {
                return Ok(choice);
            }
        }
    }

    // Strategy 3: Word-boundary search in full text — return first match found
    let mut best: Option<(&'a str, usize)> = None;

    for &choice in valid_choices {
        let choice_lower = choice.to_lowercase();
        if let Some(pos) = find_word_boundary_match(&lower, &choice_lower) {
            match best {
                None => best = Some((choice, pos)),
                Some((_, best_pos)) if pos < best_pos => best = Some((choice, pos)),
                _ => {}
            }
        }
    }

    if let Some((choice, _)) = best {
        return Ok(choice);
    }

    Err(ParseError::NoMatchingChoice {
        valid: valid_choices.iter().map(|s| s.to_string()).collect(),
    })
}

/// Find a word-boundary match of `needle` in `haystack`.
/// Returns the position of the first match, or None.
fn find_word_boundary_match(haystack: &str, needle: &str) -> Option<usize> {
    let h_bytes = haystack.as_bytes();
    let n_len = needle.len();
    let mut search_from = 0;

    while let Some(pos) = haystack[search_from..].find(needle) {
        let abs_pos = search_from + pos;
        let end_pos = abs_pos + n_len;

        // Check boundary before
        let boundary_before = abs_pos == 0 || !h_bytes[abs_pos - 1].is_ascii_alphanumeric();

        // Check boundary after
        let boundary_after = end_pos >= haystack.len() || !h_bytes[end_pos].is_ascii_alphanumeric();

        if boundary_before && boundary_after {
            return Some(abs_pos);
        }

        search_from = abs_pos + 1;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        let result = parse_choice("positive", &["positive", "negative"]).unwrap();
        assert_eq!(result, "positive");
    }

    #[test]
    fn with_period() {
        let result = parse_choice("positive.", &["positive", "negative"]).unwrap();
        assert_eq!(result, "positive");
    }

    #[test]
    fn bold() {
        let result = parse_choice("**positive**", &["positive", "negative"]).unwrap();
        assert_eq!(result, "positive");
    }

    #[test]
    fn quoted() {
        let result = parse_choice("\"positive\"", &["positive", "negative"]).unwrap();
        assert_eq!(result, "positive");
    }

    #[test]
    fn in_prose() {
        let result =
            parse_choice("I'd classify this as positive", &["positive", "negative"]).unwrap();
        assert_eq!(result, "positive");
    }

    #[test]
    fn case_insensitive() {
        let result = parse_choice("POSITIVE", &["positive", "negative"]).unwrap();
        assert_eq!(result, "positive");
    }

    #[test]
    fn first_wins() {
        let result =
            parse_choice("positive and negative aspects", &["positive", "negative"]).unwrap();
        assert_eq!(result, "positive");
    }

    #[test]
    fn with_think() {
        let result = parse_choice("<think>hmm</think>negative", &["positive", "negative"]).unwrap();
        assert_eq!(result, "negative");
    }

    #[test]
    fn no_match() {
        let result = parse_choice("maybe", &["positive", "negative"]);
        assert!(result.is_err());
    }

    #[test]
    fn no_substring() {
        let result = parse_choice("unpositive", &["positive"]);
        assert!(result.is_err());
    }
}
