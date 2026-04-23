//! Executor backed by [`OllamaChatClient`].
//!
//! Phase 1 contract: every successful model response is wrapped as
//! `Deliberation::Defer { until: Never, reasoning: response_text }`. The
//! executor does not yet parse `Act`/`Defer`/`Abandon` out of structured
//! model output — that lands once we settle on a prompt format. Until then,
//! the model's free-form reply is the deliberation's reasoning, which is
//! exactly what a human-facing `Message` flow needs (the harness returns it
//! in `PeerReply::Deliberation`).
//!
//! Out of scope for Phase 1:
//!   - Token accounting against `ctx.budget` (Phase 2 — wiring the
//!     `prompt_eval_count` / `eval_count` fields into a Reservation).
//!   - Memory recall (Phase 2 — `ctx.memory` is ignored).
//!   - Conversation history beyond the single incoming message.

use async_trait::async_trait;

use crate::bus::{EventPayload, PeerMessage};
use crate::executor::{ExecutionContext, Executor, ExecutorError};
use crate::ollama::{OllamaChatClient, OllamaError};
use crate::types::{DeferCondition, Deliberation};

#[derive(Debug, Clone)]
pub struct OllamaExecutor {
    client: OllamaChatClient,
}

impl OllamaExecutor {
    pub fn new(client: OllamaChatClient) -> Self {
        Self { client }
    }

    pub fn client(&self) -> &OllamaChatClient {
        &self.client
    }
}

#[async_trait]
impl Executor for OllamaExecutor {
    async fn deliberate(&self, ctx: ExecutionContext) -> Result<Deliberation, ExecutorError> {
        let user_message = render_user_message(&ctx.message)?;
        let system = if ctx.config.system_prompt.is_empty() {
            None
        } else {
            Some(ctx.config.system_prompt.as_str())
        };

        let response = self
            .client
            .chat(system, &user_message)
            .await
            .map_err(map_ollama_error)?;

        Ok(Deliberation::Defer {
            until: DeferCondition::Never,
            reasoning: response,
        })
    }
}

fn render_user_message(msg: &PeerMessage) -> Result<String, ExecutorError> {
    match msg {
        PeerMessage::Message { text, .. } => Ok(text.clone()),
        PeerMessage::Delegate { task, label, .. } => Ok(format!(
            "Delegated task: {label}\nCapability: {cap}\nArgs: {args}",
            cap = task.capability.0,
            args = task.args,
        )),
        PeerMessage::Event { payload, .. } => Ok(describe_event(payload)),
        PeerMessage::Ping | PeerMessage::Status => Err(ExecutorError::Internal(format!(
            "ollama executor cannot deliberate on control message: {msg:?}"
        ))),
    }
}

fn describe_event(payload: &EventPayload) -> String {
    match payload {
        EventPayload::CiFinished {
            project,
            branch,
            status,
            pipeline_url,
        } => format!(
            "CI pipeline {pipeline_url} for {project}@{branch} finished with status {status:?}"
        ),
        EventPayload::FileChanged { path, event } => {
            format!("File {path} changed: {event:?}")
        }
        EventPayload::Signal { signal } => format!("Signal received: {signal}"),
        EventPayload::Raw { json } => format!("Raw event: {json}"),
    }
}

fn map_ollama_error(err: OllamaError) -> ExecutorError {
    match err {
        OllamaError::Connection(msg) => ExecutorError::ModelUnavailable(msg),
        OllamaError::ModelNotFound(msg) => {
            ExecutorError::ModelUnavailable(format!("model not found: {msg}"))
        }
        OllamaError::Unauthorized(msg) => {
            ExecutorError::ModelUnavailable(format!("unauthorized: {msg}"))
        }
        OllamaError::Unexpected { status, body } => {
            ExecutorError::ModelUnavailable(format!("unexpected status {status}: {body}"))
        }
        OllamaError::Serde(e) => ExecutorError::InvalidResponse(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::bus::PeerOrigin;
    use crate::types::{
        AgentConfig, AgentId, Allocation, BudgetPolicy, BudgetSnapshot, CommitPolicy, MemoryScope,
        ModelProvider, ModelRef,
    };

    fn agent_config(system_prompt: &str) -> Arc<AgentConfig> {
        Arc::new(AgentConfig {
            id: AgentId::new("barnaby"),
            model: ModelRef {
                provider: ModelProvider::Ollama,
                model: "llama3:8b".into(),
            },
            system_prompt: system_prompt.into(),
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

    fn message_ctx(config: Arc<AgentConfig>, text: &str) -> ExecutionContext {
        ExecutionContext {
            config,
            message: PeerMessage::Message {
                from: PeerOrigin::External {
                    label: "test".into(),
                },
                text: text.into(),
                sent_at: chrono::Utc::now(),
            },
            budget: BudgetSnapshot {
                remaining: Allocation::zero(),
                depth: 0,
                max_depth: 3,
            },
            memory: None,
        }
    }

    #[tokio::test]
    async fn deliberate_message_returns_defer_with_response_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "I am Barnaby."},
                "done": true
            })))
            .mount(&server)
            .await;

        let exec = OllamaExecutor::new(OllamaChatClient::new(server.uri(), "llama3:8b"));
        let got = exec
            .deliberate(message_ctx(
                agent_config("You are Barnaby."),
                "who are you?",
            ))
            .await
            .unwrap();

        match got {
            Deliberation::Defer { until, reasoning } => {
                assert!(matches!(until, DeferCondition::Never));
                assert_eq!(reasoning, "I am Barnaby.");
            }
            other => panic!("expected Defer, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn deliberate_maps_404_to_model_unavailable() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(404).set_body_string("not pulled"))
            .mount(&server)
            .await;

        let exec = OllamaExecutor::new(OllamaChatClient::new(server.uri(), "missing"));
        let err = exec
            .deliberate(message_ctx(agent_config(""), "hi"))
            .await
            .unwrap_err();
        assert!(matches!(err, ExecutorError::ModelUnavailable(_)));
    }
}
