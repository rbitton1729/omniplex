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
//! Multi-turn history is held per-executor in an `Arc<Mutex<Vec<ChatMessage>>>`.
//! Each `deliberate` call prepends the system prompt, appends the accumulated
//! history, then the new user turn; on success both the user message and the
//! assistant reply are appended to history. The mutex is held across the
//! HTTP call, so concurrent deliberations on the same executor serialize —
//! that matches the single accept-loop-per-agent harness model.
//!
//! Out of scope for Phase 1:
//!   - Token accounting against `ctx.budget` (Phase 2 — wiring the
//!     `prompt_eval_count` / `eval_count` fields into a Reservation).
//!   - Memory recall (Phase 2 — `ctx.memory` is ignored).
//!   - History compaction / truncation (the vec grows unbounded for now).

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::bus::{EventPayload, PeerMessage};
use crate::executor::{ExecutionContext, Executor, ExecutorError};
use crate::ollama::{ChatMessage, OllamaChatClient, OllamaError};
use crate::types::{DeferCondition, Deliberation};

#[derive(Debug, Clone)]
pub struct OllamaExecutor {
    client: OllamaChatClient,
    history: Arc<Mutex<Vec<ChatMessage>>>,
}

impl OllamaExecutor {
    pub fn new(client: OllamaChatClient) -> Self {
        Self {
            client,
            history: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn client(&self) -> &OllamaChatClient {
        &self.client
    }
}

#[async_trait]
impl Executor for OllamaExecutor {
    async fn deliberate(&self, ctx: ExecutionContext) -> Result<Deliberation, ExecutorError> {
        let user_message = render_user_message(&ctx.message)?;

        let mut history = self.history.lock().await;
        let mut messages = Vec::with_capacity(history.len() + 2);
        if !ctx.config.system_prompt.is_empty() {
            messages.push(ChatMessage::new("system", ctx.config.system_prompt.clone()));
        }
        messages.extend(history.iter().cloned());
        messages.push(ChatMessage::new("user", user_message.clone()));

        let response = self
            .client
            .chat_messages(&ctx.config.model.model, messages)
            .await
            .map_err(map_ollama_error)?;

        history.push(ChatMessage::new("user", user_message));
        history.push(ChatMessage::new("assistant", response.clone()));

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

    use wiremock::matchers::{body_json, method, path};
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

        let exec = OllamaExecutor::new(OllamaChatClient::new(server.uri()));
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
    async fn deliberate_accumulates_history_across_turns() {
        let server = MockServer::start().await;

        // First turn: system + user("hello").
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_json(serde_json::json!({
                "model": "llama3:8b",
                "messages": [
                    {"role": "system", "content": "You are Barnaby."},
                    {"role": "user", "content": "hello"}
                ],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "hi there"},
                "done": true
            })))
            .expect(1)
            .mount(&server)
            .await;

        // Second turn: system + full prior history + new user("still there?").
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_json(serde_json::json!({
                "model": "llama3:8b",
                "messages": [
                    {"role": "system", "content": "You are Barnaby."},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                    {"role": "user", "content": "still there?"}
                ],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "yes"},
                "done": true
            })))
            .expect(1)
            .mount(&server)
            .await;

        let exec = OllamaExecutor::new(OllamaChatClient::new(server.uri()));
        let cfg = agent_config("You are Barnaby.");

        let first = exec
            .deliberate(message_ctx(cfg.clone(), "hello"))
            .await
            .unwrap();
        match first {
            Deliberation::Defer { reasoning, .. } => assert_eq!(reasoning, "hi there"),
            other => panic!("expected Defer, got {other:?}"),
        }

        let second = exec
            .deliberate(message_ctx(cfg, "still there?"))
            .await
            .unwrap();
        match second {
            Deliberation::Defer { reasoning, .. } => assert_eq!(reasoning, "yes"),
            other => panic!("expected Defer, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn deliberate_failed_turn_does_not_poison_history() {
        let server = MockServer::start().await;

        // First call fails with 503 — history must NOT grow.
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_json(serde_json::json!({
                "model": "llama3:8b",
                "messages": [
                    {"role": "user", "content": "first"}
                ],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(503).set_body_string("overloaded"))
            .expect(1)
            .mount(&server)
            .await;

        // Second call must show only the new user message, not a stale pair.
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_json(serde_json::json!({
                "model": "llama3:8b",
                "messages": [
                    {"role": "user", "content": "second"}
                ],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "ok"},
                "done": true
            })))
            .expect(1)
            .mount(&server)
            .await;

        let exec = OllamaExecutor::new(OllamaChatClient::new(server.uri()));
        let cfg = agent_config("");

        let err = exec
            .deliberate(message_ctx(cfg.clone(), "first"))
            .await
            .unwrap_err();
        assert!(matches!(err, ExecutorError::ModelUnavailable(_)));

        let ok = exec.deliberate(message_ctx(cfg, "second")).await.unwrap();
        match ok {
            Deliberation::Defer { reasoning, .. } => assert_eq!(reasoning, "ok"),
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

        let exec = OllamaExecutor::new(OllamaChatClient::new(server.uri()));
        let err = exec
            .deliberate(message_ctx(agent_config(""), "hi"))
            .await
            .unwrap_err();
        assert!(matches!(err, ExecutorError::ModelUnavailable(_)));
    }
}
