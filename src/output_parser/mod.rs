//! # LLM Output Parser
//!
//! Production-grade parser for extracting structured data from LLM responses.
//! Handles think blocks, markdown fences, malformed JSON, and other real-world
//! model output patterns without requiring an additional LLM call.
//!
//! ## Parsers Available
//!
//! | Parser | Use Case |
//! |--------|----------|
//! | [`parse_json`] | Extract typed JSON structs |
//! | [`parse_json_value`] | Extract untyped JSON |
//! | [`parse_string_list`] | Extract cleaned string lists (tags, items) |
//! | [`parse_string_list_raw`] | Extract string lists without cleaning |
//! | [`parse_xml_tag`] | Extract content from an XML tag |
//! | [`parse_xml_tags`] | Extract content from multiple XML tags |
//! | [`parse_choice`] | Extract a choice from valid options |
//! | [`parse_number`] | Extract a numeric value |
//! | [`parse_number_in_range`] | Extract a bounded numeric value |
//! | [`parse_text`] | Clean text extraction |
//! | `parse_yaml` | Extract typed YAML (feature: `yaml`) |
//!
//! ## Shared Utilities
//!
//! | Function | Purpose |
//! |----------|---------|
//! | [`strip_think_tags`] | Remove `<think>` blocks from text |
//! | [`try_repair_json`] | Fix common LLM JSON errors |

pub mod choice;
pub mod error;
pub mod extract;
pub mod json;
pub mod list;
pub mod number;
pub mod repair;
pub mod streaming;
pub mod text;
pub mod xml;

#[cfg(feature = "yaml")]
pub mod yaml;

// Re-export all public functions at module level
pub use choice::parse_choice;
pub use error::ParseError;
pub use extract::{preprocess, strip_think_tags};
pub use json::{parse_json, parse_json_value};
pub use list::{parse_string_list, parse_string_list_raw};
pub use number::{parse_number, parse_number_in_range};
pub use repair::try_repair_json;
pub use text::parse_text;
pub use xml::{parse_xml_tag, parse_xml_tags};

#[cfg(feature = "yaml")]
pub use yaml::parse_yaml;
