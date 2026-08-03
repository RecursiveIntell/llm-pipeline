#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostModel {
    /// Cost per 1 million input (prompt) tokens.
    pub input_cost_per_million_tokens: f64,
    /// Cost per 1 million output (completion) tokens.
    pub output_cost_per_million_tokens: f64,
    /// Cost per 1 million cached input tokens, if cheaper.
    pub cache_hit_cost_per_million_tokens: f64,
}

impl CostModel {
    /// Create a cost model with the given per-million-token prices.
    pub fn new(input: f64, output: f64, cache_hit: f64) -> Self {
        Self {
            input_cost_per_million_tokens: input,
            output_cost_per_million_tokens: output,
            cache_hit_cost_per_million_tokens: cache_hit,
        }
    }

    /// Estimate cost for a token-usage snapshot.
    pub fn estimate(&self, usage: &crate::payload::TokenUsage, cache_hit: bool) -> f64 {
        let prompt_tokens = usage.prompt_tokens as f64;
        let completion_tokens = usage.completion_tokens as f64;
        let input_cost = if cache_hit {
            prompt_tokens * self.cache_hit_cost_per_million_tokens / 1_000_000.0
        } else {
            prompt_tokens * self.input_cost_per_million_tokens / 1_000_000.0
        };
        let output_cost = completion_tokens * self.output_cost_per_million_tokens / 1_000_000.0;
        input_cost + output_cost
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            input_cost_per_million_tokens: 0.0,
            output_cost_per_million_tokens: 0.0,
            cache_hit_cost_per_million_tokens: 0.0,
        }
    }
}
