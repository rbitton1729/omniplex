//! Deliberation executor: the model-call abstraction.
//!
//! The executor is the one place that turns a `PeerMessage` into a
//! `Deliberation`. Phase 1 ships only the trait, its context type, and a
//! stub; the real model client (Anthropic / Ollama / OpenAI-compatible)
//! lands in a follow-up. The harness and daemon already take an
//! `Arc<dyn Executor>` so swapping the stub for a production impl is a
//! single call-site change in `omniplexd`.
//!
//! Scope: no I/O, no HTTP client, no token accounting. Those land with the
//! real impl. `StubExecutor` exists so tests and local dev can exercise
//! the harness end-to-end without reaching a provider.

use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;

use crate::bus::PeerMessage;
use crate::memory::MemoryService;
use crate::ollama::OllamaChatClient;
use crate::ollama_executor::OllamaExecutor;
use crate::types::{
    AgentConfig, BudgetExhausted, BudgetSnapshot, DeferCondition, Deliberation, ModelProvider,
    ModelRef,
};

#[derive(Debug, Error)]
pub enum ExecutorError {
    #[error("budget exhausted: {0}")]
    BudgetExhausted(#[from] BudgetExhausted),
    #[error("model unavailable: {0}")]
    ModelUnavailable(String),
    #[error("invalid model response: {0}")]
    InvalidResponse(String),
    #[error("internal executor error: {0}")]
    Internal(String),
}

/// Everything a deliberation needs: the agent's config, the incoming
/// message (already parsed off the bus), a budget snapshot the executor
/// may reserve against, and an optional memory service for recall.
/// Cheap to construct — the heavy fields are Arc-shared.
pub struct ExecutionContext {
    pub config: Arc<AgentConfig>,
    pub message: PeerMessage,
    pub budget: BudgetSnapshot,
    pub memory: Option<Arc<dyn MemoryService>>,
}

#[async_trait]
pub trait Executor: Send + Sync + std::fmt::Debug {
    async fn deliberate(&self, ctx: ExecutionContext) -> Result<Deliberation, ExecutorError>;
}

/// Always defers with `DeferCondition::Never`. Placeholder until the real
/// model client lands; also useful in tests that want end-to-end harness
/// coverage without hitting a provider.
#[derive(Debug, Default, Clone, Copy)]
pub struct StubExecutor;

#[async_trait]
impl Executor for StubExecutor {
    async fn deliberate(&self, _ctx: ExecutionContext) -> Result<Deliberation, ExecutorError> {
        Ok(Deliberation::Defer {
            until: DeferCondition::Never,
            reasoning: "executor not yet wired to model client".into(),
        })
    }
}

/// Pick a concrete [`Executor`] for a given model reference.
///
/// Phase 1 wiring: `Ollama` returns a real `OllamaExecutor` pointed at the
/// local daemon; everything else falls back to `StubExecutor`. The Anthropic
/// path will land once the API client does.
pub fn executor_for(model: &ModelRef) -> Arc<dyn Executor> {
    match model.provider {
        ModelProvider::Ollama => Arc::new(OllamaExecutor::new(OllamaChatClient::new(
            "http://localhost:11434",
        ))),
        ModelProvider::Anthropic | ModelProvider::OpenAiCompatible => Arc::new(StubExecutor),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::PeerOrigin;
    use crate::types::{
        AgentId, Allocation, BudgetPolicy, CommitPolicy, MemoryScope, ModelProvider, ModelRef,
    };

    fn agent_config() -> Arc<AgentConfig> {
        Arc::new(AgentConfig {
            id: AgentId::new("barnaby"),
            model: ModelRef {
                provider: ModelProvider::Anthropic,
                model: "claude-opus-4-7".into(),
            },
            system_prompt: "You are Barnaby.".into(),
            memory_scope: MemoryScope::Private,
            allowed_capabilities: Vec::new(),
            default_budget: BudgetPolicy {
                input_tokens: 1_000,
                output_tokens: 1_000,
                calls: 10,
                max_depth: 2,
            },
            commit_policy: CommitPolicy::Manual,
        })
    }

    #[tokio::test]
    async fn executor_stub_always_defers() {
        let ctx = ExecutionContext {
            config: agent_config(),
            message: PeerMessage::Message {
                from: PeerOrigin::External {
                    label: "test".into(),
                },
                text: "hello".into(),
                sent_at: chrono::Utc::now(),
            },
            budget: BudgetSnapshot {
                remaining: Allocation::zero(),
                depth: 0,
                max_depth: 3,
            },
            memory: None,
        };

        let got = StubExecutor.deliberate(ctx).await.unwrap();
        match got {
            Deliberation::Defer { until, reasoning } => {
                assert!(matches!(until, DeferCondition::Never));
                assert_eq!(reasoning, "executor not yet wired to model client");
            }
            other => panic!("expected Defer, got {other:?}"),
        }
    }
}
