//! ModelCatalog: pricing and context-window lookup, keyed by (provider, model).
//!
//! AgentConfigs only carry a ModelRef (provider + model name); the catalog is
//! the single place where pricing and context limits live. When Anthropic
//! tweaks prices or publishes a new model, we update the catalog, not every
//! agent TOML.
//!
//! The catalog is constructed at daemon startup — typically from a
//! `models.toml` next to the main config, with [`ModelCatalog::with_builtin`]
//! as a reasonable default for dev.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::{ModelProvider, ModelRef};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub max_context: usize,
    /// USD per input token. 0.0 for local models.
    pub cost_per_input_token: f64,
    /// USD per output token.
    pub cost_per_output_token: f64,
}

#[derive(Debug, Error)]
pub enum CatalogError {
    #[error("unknown model: {0:?} / {1}")]
    UnknownModel(ModelProvider, String),
}

#[derive(Debug, Clone, Default)]
pub struct ModelCatalog {
    entries: HashMap<(ModelProvider, String), ModelEntry>,
}

impl ModelCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, provider: ModelProvider, model: impl Into<String>, entry: ModelEntry) {
        self.entries.insert((provider, model.into()), entry);
    }

    pub fn lookup(&self, model: &ModelRef) -> Result<&ModelEntry, CatalogError> {
        self.entries
            .get(&(model.provider, model.model.clone()))
            .ok_or_else(|| CatalogError::UnknownModel(model.provider, model.model.clone()))
    }

    /// Built-in defaults for common models. Hand-maintained; update on
    /// provider price changes. Local Ollama models are priced at zero since
    /// the cost is hardware amortization, not per-token.
    pub fn with_builtin() -> Self {
        let mut c = Self::new();

        // Anthropic — prices in USD per token (not per million).
        c.insert(
            ModelProvider::Anthropic,
            "claude-opus-4-7",
            ModelEntry {
                max_context: 1_000_000,
                cost_per_input_token: 15.0 / 1_000_000.0,
                cost_per_output_token: 75.0 / 1_000_000.0,
            },
        );
        c.insert(
            ModelProvider::Anthropic,
            "claude-sonnet-4-6",
            ModelEntry {
                max_context: 200_000,
                cost_per_input_token: 3.0 / 1_000_000.0,
                cost_per_output_token: 15.0 / 1_000_000.0,
            },
        );
        c.insert(
            ModelProvider::Anthropic,
            "claude-haiku-4-5",
            ModelEntry {
                max_context: 200_000,
                cost_per_input_token: 1.0 / 1_000_000.0,
                cost_per_output_token: 5.0 / 1_000_000.0,
            },
        );

        // Ollama — free. Context windows depend on the running server config.
        c.insert(
            ModelProvider::Ollama,
            "qwen3:32b",
            ModelEntry { max_context: 32_768, cost_per_input_token: 0.0, cost_per_output_token: 0.0 },
        );
        c.insert(
            ModelProvider::Ollama,
            "nomic-embed-text",
            ModelEntry { max_context: 8_192, cost_per_input_token: 0.0, cost_per_output_token: 0.0 },
        );

        c
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_has_opus_and_sonnet() {
        let c = ModelCatalog::with_builtin();
        c.lookup(&ModelRef {
            provider: ModelProvider::Anthropic,
            model: "claude-opus-4-7".into(),
        })
        .unwrap();
        c.lookup(&ModelRef {
            provider: ModelProvider::Anthropic,
            model: "claude-sonnet-4-6".into(),
        })
        .unwrap();
    }

    #[test]
    fn unknown_model_errors() {
        let c = ModelCatalog::with_builtin();
        let err = c
            .lookup(&ModelRef {
                provider: ModelProvider::Anthropic,
                model: "claude-ghost-99".into(),
            })
            .unwrap_err();
        assert!(matches!(err, CatalogError::UnknownModel(_, _)));
    }
}
