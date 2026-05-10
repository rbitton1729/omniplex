//! Executor backed by [`OllamaChatClient`].
//!
//! Contract: the model is asked to reply with a single JSON object that
//! deserializes into [`Deliberation`] (`Act` / `Defer` / `Abandon`). The
//! prompt addendum in [`FORMAT_CONTRACT`] pins the schema, and the executor
//! parses the reply with `serde_json`. A reply that fails to parse surfaces
//! as `ExecutorError::InvalidResponse` — the executor never silently rewrites
//! malformed output as a Defer.
//!
//! Multi-turn history is held per-executor in an `Arc<Mutex<Vec<ChatMessage>>>`.
//! Each `deliberate` call prepends the system prompt (user prompt joined with
//! [`FORMAT_CONTRACT`]), appends the accumulated history, then the new user
//! turn; on success both the user message and the raw assistant JSON are
//! appended to history so the model sees its prior structured replies. The
//! mutex is held across the HTTP call, so concurrent deliberations on the
//! same executor serialize — that matches the single accept-loop-per-agent
//! harness model. Failed turns (HTTP error or malformed reply) do not mutate
//! history.
//!
//! Out of scope here:
//!   - Token accounting against `ctx.budget` (Phase 2 — wiring the
//!     `prompt_eval_count` / `eval_count` fields into a Reservation).
//!   - History compaction / truncation (the vec grows unbounded for now).

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::bus::{EventPayload, PeerMessage};
use crate::executor::{ExecutionContext, Executor, ExecutorError};
use crate::ollama::{ChatMessage, OllamaChatClient, OllamaError};
use crate::types::Deliberation;

/// Strict response contract appended to the agent's system prompt. The model
/// must reply with a single JSON object that round-trips through
/// `serde_json::from_str::<Deliberation>` — anything else surfaces as
/// [`ExecutorError::InvalidResponse`].
pub(crate) const FORMAT_CONTRACT: &str = r#"Reply with EXACTLY one JSON object that matches one of the following schemas. Output only the JSON — no Markdown, no prose, no code fences.

1) Call a capability:
   {"kind":"act","task":{"capability":"<id>","args":{...},"idempotency_key":"<unique-string>"},"reasoning":"<why>","next_check":"never"}

2) Wait without acting:
   {"kind":"defer","until":"never","reasoning":"<why>"}

3) Give up:
   {"kind":"abandon","reasoning":"<why>"}

For "until" or "next_check" you may use the string "never", or {"time":"<RFC3339 timestamp>"}, or {"event":"<filter>"}."#;

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
        let user_message = augment_with_recall(&ctx, user_message).await;

        let mut history = self.history.lock().await;
        let mut messages = Vec::with_capacity(history.len() + 2);
        messages.push(ChatMessage::new(
            "system",
            build_system_prompt(&ctx.config.system_prompt),
        ));
        messages.extend(history.iter().cloned());
        messages.push(ChatMessage::new("user", user_message.clone()));

        let response = self
            .client
            .chat_messages(&ctx.config.model.model, messages)
            .await
            .map_err(map_ollama_error)?;

        let deliberation = parse_deliberation(&response)?;

        if let Some(memory) = &ctx.memory {
            let turn = crate::memory::ActivityTurn {
                goal_id: uuid::Uuid::new_v4(),
                turn_id: uuid::Uuid::new_v4(),
                entries: vec![
                    crate::memory::TurnEntry {
                        role: "user".into(),
                        content: user_message.clone(),
                        timestamp: chrono::Utc::now(),
                        tool_name: None,
                    },
                    crate::memory::TurnEntry {
                        role: "assistant".into(),
                        content: response.clone(),
                        timestamp: chrono::Utc::now(),
                        tool_name: None,
                    },
                ],
            };
            memory
                .record_turn(&ctx.config.id, turn)
                .await
                .map_err(|e| ExecutorError::Internal(format!("record turn: {e}")))?;
        }

        history.push(ChatMessage::new("user", user_message));
        history.push(ChatMessage::new("assistant", response));

        Ok(deliberation)
    }
}

fn build_system_prompt(user_prompt: &str) -> String {
    if user_prompt.is_empty() {
        FORMAT_CONTRACT.to_string()
    } else {
        format!("{user_prompt}\n\n{FORMAT_CONTRACT}")
    }
}

fn parse_deliberation(raw: &str) -> Result<Deliberation, ExecutorError> {
    serde_json::from_str::<Deliberation>(raw.trim()).map_err(|e| {
        ExecutorError::InvalidResponse(format!(
            "model output was not a Deliberation JSON object: {e}; raw={raw:?}"
        ))
    })
}

async fn augment_with_recall(ctx: &ExecutionContext, user_message: String) -> String {
    let Some(memory) = &ctx.memory else {
        return user_message;
    };

    let mut opts = crate::memory::SearchOpts::default();
    opts.limit = 3;

    let Ok(hits) = memory.recall(&ctx.config.id, &user_message, opts).await else {
        return user_message;
    };
    if hits.is_empty() {
        return user_message;
    }

    let context_lines = hits
        .into_iter()
        .map(|hit| format!("- {}", hit.snippet.trim()))
        .collect::<Vec<_>>()
        .join("\n");

    format!("User message:\n{user_message}\n\nRelevant prior context:\n{context_lines}")
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
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::bus::PeerOrigin;
    use crate::memory::{Hit, MemoryError, MemoryService, SearchOpts};
    use crate::types::{
        AgentConfig, AgentId, Allocation, BudgetPolicy, BudgetSnapshot, CapabilityId, CommitPolicy,
        DeferCondition, MemoryScope, ModelProvider, ModelRef, Task,
    };

    fn defer_json(reasoning: &str) -> String {
        serde_json::to_string(&Deliberation::Defer {
            until: DeferCondition::Never,
            reasoning: reasoning.into(),
        })
        .expect("serialize defer")
    }

    fn abandon_json(reasoning: &str) -> String {
        serde_json::to_string(&Deliberation::Abandon {
            reasoning: reasoning.into(),
        })
        .expect("serialize abandon")
    }

    fn act_json(capability: &str, args: serde_json::Value, idempotency_key: &str) -> String {
        serde_json::to_string(&Deliberation::Act {
            task: Task {
                capability: CapabilityId(capability.into()),
                args,
                idempotency_key: idempotency_key.into(),
            },
            reasoning: "let's do it".into(),
            next_check: DeferCondition::Never,
        })
        .expect("serialize act")
    }

    fn rendered_system_prompt(user_prompt: &str) -> String {
        super::build_system_prompt(user_prompt)
    }

    #[derive(Debug)]
    struct StaticMemory {
        hits: Vec<Hit>,
    }

    #[async_trait]
    impl MemoryService for StaticMemory {
        async fn record_turn(
            &self,
            _agent: &AgentId,
            _turn: crate::memory::ActivityTurn,
        ) -> Result<(), MemoryError> {
            Ok(())
        }

        async fn search(
            &self,
            _agent: &AgentId,
            _query: &str,
            _opts: SearchOpts,
        ) -> Result<Vec<Hit>, MemoryError> {
            Ok(self.hits.clone())
        }

        async fn recall(
            &self,
            _agent: &AgentId,
            _context: &str,
            _opts: SearchOpts,
        ) -> Result<Vec<Hit>, MemoryError> {
            Ok(self.hits.clone())
        }
    }

    #[derive(Debug, Default)]
    struct RecordingMemory {
        turns: Mutex<Vec<crate::memory::ActivityTurn>>,
    }

    #[async_trait]
    impl MemoryService for RecordingMemory {
        async fn record_turn(
            &self,
            _agent: &AgentId,
            turn: crate::memory::ActivityTurn,
        ) -> Result<(), MemoryError> {
            self.turns
                .lock()
                .expect("recording memory poisoned")
                .push(turn);
            Ok(())
        }

        async fn search(
            &self,
            _agent: &AgentId,
            _query: &str,
            _opts: SearchOpts,
        ) -> Result<Vec<Hit>, MemoryError> {
            Ok(Vec::new())
        }

        async fn recall(
            &self,
            _agent: &AgentId,
            _context: &str,
            _opts: SearchOpts,
        ) -> Result<Vec<Hit>, MemoryError> {
            Ok(Vec::new())
        }
    }

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
    async fn deliberate_parses_structured_defer() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": defer_json("I am Barnaby.")},
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
    async fn deliberate_parses_structured_abandon() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": abandon_json("nothing more to do")},
                "done": true
            })))
            .mount(&server)
            .await;

        let exec = OllamaExecutor::new(OllamaChatClient::new(server.uri()));
        let got = exec
            .deliberate(message_ctx(agent_config(""), "anything?"))
            .await
            .unwrap();

        match got {
            Deliberation::Abandon { reasoning } => assert_eq!(reasoning, "nothing more to do"),
            other => panic!("expected Abandon, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn deliberate_parses_structured_act() {
        let server = MockServer::start().await;
        let args = serde_json::json!({"to": "diogenes", "text": "ping"});
        let body = act_json("send_message", args.clone(), "msg-123");
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": body},
                "done": true
            })))
            .mount(&server)
            .await;

        let exec = OllamaExecutor::new(OllamaChatClient::new(server.uri()));
        let got = exec
            .deliberate(message_ctx(agent_config(""), "say hi to diogenes"))
            .await
            .unwrap();

        match got {
            Deliberation::Act {
                task,
                reasoning,
                next_check,
            } => {
                assert_eq!(task.capability.0, "send_message");
                assert_eq!(task.args, args);
                assert_eq!(task.idempotency_key, "msg-123");
                assert_eq!(reasoning, "let's do it");
                assert!(matches!(next_check, DeferCondition::Never));
            }
            other => panic!("expected Act, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn deliberate_malformed_output_is_invalid_response() {
        let server = MockServer::start().await;
        let sys = rendered_system_prompt("");

        // First turn: model returns plain text — must surface as InvalidResponse.
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_json(serde_json::json!({
                "model": "llama3:8b",
                "messages": [
                    {"role": "system", "content": sys.clone()},
                    {"role": "user", "content": "first"}
                ],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "hello, I am not JSON"},
                "done": true
            })))
            .expect(1)
            .mount(&server)
            .await;

        // Second turn: history must NOT include the malformed reply — the
        // request body matches an empty history with just the new user turn.
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_json(serde_json::json!({
                "model": "llama3:8b",
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": "second"}
                ],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": defer_json("ok")},
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
        assert!(
            matches!(err, ExecutorError::InvalidResponse(_)),
            "got {err:?}"
        );
        let _ = exec
            .deliberate(message_ctx(cfg, "second"))
            .await
            .expect("second turn parses");
    }

    #[tokio::test]
    async fn deliberate_accumulates_history_across_turns() {
        let server = MockServer::start().await;
        let sys = rendered_system_prompt("You are Barnaby.");
        let first_reply = defer_json("hi there");
        let second_reply = defer_json("yes");

        // First turn: system + user("hello").
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_json(serde_json::json!({
                "model": "llama3:8b",
                "messages": [
                    {"role": "system", "content": sys.clone()},
                    {"role": "user", "content": "hello"}
                ],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": first_reply.clone()},
                "done": true
            })))
            .expect(1)
            .mount(&server)
            .await;

        // Second turn: system + full prior history (with the raw structured
        // assistant JSON) + new user("still there?").
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_json(serde_json::json!({
                "model": "llama3:8b",
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": first_reply},
                    {"role": "user", "content": "still there?"}
                ],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": second_reply},
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
    async fn deliberate_includes_recalled_context_in_user_prompt() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_json(serde_json::json!({
                "model": "llama3:8b",
                "messages": [
                    {"role": "system", "content": rendered_system_prompt("")},
                    {"role": "user", "content": "User message:\nwhat happened with pipeline seven?\n\nRelevant prior context:\n- Three CI failures on pipeline seven in the last hour."}
                ],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": defer_json("I found the prior failures.")},
                "done": true
            })))
            .expect(1)
            .mount(&server)
            .await;

        let exec = OllamaExecutor::new(OllamaChatClient::new(server.uri()));
        let ctx = ExecutionContext {
            config: agent_config(""),
            message: PeerMessage::Message {
                from: PeerOrigin::External {
                    label: "test".into(),
                },
                text: "what happened with pipeline seven?".into(),
                sent_at: chrono::Utc::now(),
            },
            budget: BudgetSnapshot {
                remaining: Allocation::zero(),
                depth: 0,
                max_depth: 3,
            },
            memory: Some(Arc::new(StaticMemory {
                hits: vec![Hit {
                    uri: "agent://barnaby/goal/g/turn/t".into(),
                    ordinal: 0,
                    text: "Three CI failures on pipeline seven in the last hour.".into(),
                    snippet: "Three CI failures on pipeline seven in the last hour.".into(),
                    score: 1.0,
                    role: Some("note".into()),
                    session_id: None,
                    turn_id: None,
                }],
            })),
        };

        let got = exec.deliberate(ctx).await.unwrap();
        match got {
            Deliberation::Defer { until, reasoning } => {
                assert!(matches!(until, DeferCondition::Never));
                assert_eq!(reasoning, "I found the prior failures.");
            }
            other => panic!("expected Defer, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn deliberate_records_turn_when_memory_is_available() {
        let server = MockServer::start().await;
        let reply = defer_json("I found the prior failures.");
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": reply.clone()},
                "done": true
            })))
            .expect(1)
            .mount(&server)
            .await;

        let exec = OllamaExecutor::new(OllamaChatClient::new(server.uri()));
        let memory = Arc::new(RecordingMemory::default());
        let ctx = ExecutionContext {
            config: agent_config(""),
            message: PeerMessage::Message {
                from: PeerOrigin::External {
                    label: "test".into(),
                },
                text: "what happened with pipeline seven?".into(),
                sent_at: chrono::Utc::now(),
            },
            budget: BudgetSnapshot {
                remaining: Allocation::zero(),
                depth: 0,
                max_depth: 3,
            },
            memory: Some(memory.clone()),
        };

        let got = exec.deliberate(ctx).await.unwrap();
        match got {
            Deliberation::Defer { until, reasoning } => {
                assert!(matches!(until, DeferCondition::Never));
                assert_eq!(reasoning, "I found the prior failures.");
            }
            other => panic!("expected Defer, got {other:?}"),
        }

        let turns = memory.turns.lock().expect("recording memory poisoned");
        assert_eq!(turns.len(), 1, "executor should persist one turn");
        assert_eq!(
            turns[0].entries.len(),
            2,
            "turn should capture request and reply"
        );
        assert_eq!(turns[0].entries[0].role, "user");
        assert!(
            turns[0].entries[0]
                .content
                .contains("what happened with pipeline seven?")
        );
        assert_eq!(turns[0].entries[1].role, "assistant");
        assert_eq!(turns[0].entries[1].content, reply);
    }

    #[tokio::test]
    async fn deliberate_failed_turn_does_not_poison_history() {
        let server = MockServer::start().await;
        let sys = rendered_system_prompt("");

        // First call fails with 503 — history must NOT grow.
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_json(serde_json::json!({
                "model": "llama3:8b",
                "messages": [
                    {"role": "system", "content": sys.clone()},
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
                    {"role": "system", "content": sys},
                    {"role": "user", "content": "second"}
                ],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": defer_json("ok")},
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
