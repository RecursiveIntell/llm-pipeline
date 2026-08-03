use serde_json::Value;
#[derive(Debug, Clone, PartialEq, Default)]
pub enum GenerationConstraint {
    #[default]
    None,
    JsonSchema(Value),
    Grammar(String),
    Regex(String),
}
