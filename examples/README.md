# Provider and budget examples

The examples in this directory show the basic Ollama call (`basic_pipeline`).
Enable `--features anthropic` for Anthropic-backed calls (construct an
`AnthropicBackend` with an API key). `ExecCtx::builder(url).with_token_budget(4096)`
adds an atomic per-context token budget; `CostModel` can be supplied with
`with_cost_model`. `RetryConfig::new(2).best_of_n(3, vec![])` selects the first
successful JSON parse among concurrent candidates.

Provider examples require a live provider endpoint and credentials, so they are
intentionally documented rather than run by default.
