//! Agent harness: per-agent Unix socket accept loop.
//!
//! Each harness owns one `UnixListener` at `<socket_dir>/<name>.sock` and
//! services one `PeerMessage` per accepted connection. The reply is written
//! on the same stream and the connection is closed — matching the one-shot
//! request/response contract described in `bus.rs`.
//!
//! Routing:
//!   - `Ping`    → `Ack`
//!   - `Status`  → `Status { agent, budget-from-policy, uptime }`
//!   - `Message` / `Delegate` / `Event` → if an executor is configured,
//!     dispatch to `Executor::deliberate` and wrap its result in
//!     `PeerReply::Deliberation`; otherwise reply with a stock
//!     `Defer { Never, "no executor configured" }`.
//!
//! The harness owns no budget state itself; charging is the executor's job.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufStream};
use tokio::net::{UnixListener, UnixStream};

use crate::bus::{PeerMessage, PeerReply};
use crate::executor::{ExecutionContext, Executor};
use crate::types::{AgentConfig, BudgetSnapshot, DeferCondition, Deliberation};

#[derive(Debug, Error)]
pub enum HarnessError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}

pub struct AgentHarness {
    config: Arc<AgentConfig>,
    listener: UnixListener,
    socket_path: PathBuf,
    started: Instant,
    executor: Option<Arc<dyn Executor>>,
}

impl AgentHarness {
    /// Bind `socket_path` and prepare the accept loop. Caller is responsible
    /// for ensuring the parent directory exists and the path is free (stale
    /// sockets from a previous run must be unlinked first — `UnixListener`
    /// will otherwise fail with `AddrInUse`).
    ///
    /// `executor` is optional so tests and minimal harness fixtures can run
    /// without one; production callers should always pass `Some(...)`.
    pub fn bind(
        config: AgentConfig,
        socket_path: PathBuf,
        executor: Option<Arc<dyn Executor>>,
    ) -> Result<Self, HarnessError> {
        let listener = UnixListener::bind(&socket_path)?;
        Ok(Self {
            config: Arc::new(config),
            listener,
            socket_path,
            started: Instant::now(),
            executor,
        })
    }

    pub fn agent_name(&self) -> &str {
        &self.config.id.name
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Run the accept loop until the listener returns an error. Each
    /// connection is handled on a detached task; errors are logged to
    /// stderr (no structured logging yet).
    pub async fn run(self) -> Result<(), HarnessError> {
        let agent_name = self.config.id.name.clone();
        loop {
            let (stream, _addr) = self.listener.accept().await?;
            let cfg = self.config.clone();
            let executor = self.executor.clone();
            let started = self.started;
            let name = agent_name.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_connection(cfg, executor, started, stream).await {
                    eprintln!("harness[{name}]: connection error: {e}");
                }
            });
        }
    }
}

impl Drop for AgentHarness {
    fn drop(&mut self) {
        // Best-effort cleanup of the bound socket file. If the daemon is
        // restarted without this running (hard crash), `bind()` callers must
        // unlink stale sockets themselves.
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

async fn handle_connection(
    cfg: Arc<AgentConfig>,
    executor: Option<Arc<dyn Executor>>,
    started: Instant,
    stream: UnixStream,
) -> Result<(), HarnessError> {
    let mut framed = BufStream::new(stream);
    let mut line = String::new();
    let n = framed.read_line(&mut line).await?;
    if n == 0 {
        return Ok(());
    }

    let reply = match serde_json::from_str::<PeerMessage>(line.trim_end()) {
        Ok(msg) => route(&cfg, executor.as_ref(), started, msg).await,
        Err(e) => PeerReply::Error {
            reason: format!("invalid peer message: {e}"),
        },
    };

    let json = serde_json::to_vec(&reply)?;
    framed.write_all(&json).await?;
    framed.write_all(b"\n").await?;
    framed.flush().await?;
    Ok(())
}

async fn route(
    cfg: &Arc<AgentConfig>,
    executor: Option<&Arc<dyn Executor>>,
    started: Instant,
    msg: PeerMessage,
) -> PeerReply {
    match msg {
        PeerMessage::Ping => PeerReply::Ack,
        PeerMessage::Status => PeerReply::Status {
            agent: cfg.id.clone(),
            budget: default_budget_snapshot(cfg),
            uptime_secs: started.elapsed().as_secs(),
        },
        msg @ (PeerMessage::Message { .. }
        | PeerMessage::Delegate { .. }
        | PeerMessage::Event { .. }) => deliberate(cfg, executor, msg).await,
    }
}

async fn deliberate(
    cfg: &Arc<AgentConfig>,
    executor: Option<&Arc<dyn Executor>>,
    msg: PeerMessage,
) -> PeerReply {
    let Some(exec) = executor else {
        return PeerReply::Deliberation(Deliberation::Defer {
            until: DeferCondition::Never,
            reasoning: format!("{}: no executor configured", cfg.id.name),
        });
    };

    let budget = message_budget(cfg, &msg);
    let ctx = ExecutionContext {
        config: cfg.clone(),
        message: msg,
        budget,
        memory: None,
    };

    match exec.deliberate(ctx).await {
        Ok(delib) => PeerReply::Deliberation(delib),
        Err(e) => PeerReply::Error {
            reason: format!("executor: {e}"),
        },
    }
}

fn default_budget_snapshot(cfg: &AgentConfig) -> BudgetSnapshot {
    BudgetSnapshot {
        remaining: cfg.default_budget.allocation(),
        depth: 0,
        max_depth: cfg.default_budget.max_depth,
    }
}

fn message_budget(cfg: &AgentConfig, msg: &PeerMessage) -> BudgetSnapshot {
    match msg {
        PeerMessage::Delegate { budget, .. } => *budget,
        _ => default_budget_snapshot(cfg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::PeerOrigin;
    use crate::executor::StubExecutor;
    use crate::types::{AgentId, BudgetPolicy, CommitPolicy, MemoryScope, ModelProvider, ModelRef};

    fn cfg() -> Arc<AgentConfig> {
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
                max_depth: 3,
            },
            commit_policy: CommitPolicy::Manual,
        })
    }

    fn message() -> PeerMessage {
        PeerMessage::Message {
            from: PeerOrigin::External { label: "t".into() },
            text: "hello".into(),
            sent_at: chrono::Utc::now(),
        }
    }

    #[tokio::test]
    async fn route_ping_returns_ack() {
        let reply = route(&cfg(), None, Instant::now(), PeerMessage::Ping).await;
        assert!(matches!(reply, PeerReply::Ack));
    }

    #[tokio::test]
    async fn route_status_returns_status_without_invoking_executor() {
        let exec: Arc<dyn Executor> = Arc::new(StubExecutor);
        let reply = route(&cfg(), Some(&exec), Instant::now(), PeerMessage::Status).await;
        match reply {
            PeerReply::Status { agent, .. } => assert_eq!(agent.name, "barnaby"),
            other => panic!("expected Status, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn route_message_without_executor_defers_with_stock_reasoning() {
        let reply = route(&cfg(), None, Instant::now(), message()).await;
        match reply {
            PeerReply::Deliberation(Deliberation::Defer { until, reasoning }) => {
                assert!(matches!(until, DeferCondition::Never));
                assert!(
                    reasoning.contains("no executor configured"),
                    "unexpected reasoning: {reasoning}"
                );
            }
            other => panic!("expected Defer, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn route_message_with_stub_executor_defers_via_executor() {
        let exec: Arc<dyn Executor> = Arc::new(StubExecutor);
        let reply = route(&cfg(), Some(&exec), Instant::now(), message()).await;
        match reply {
            PeerReply::Deliberation(Deliberation::Defer { until, reasoning }) => {
                assert!(matches!(until, DeferCondition::Never));
                assert_eq!(reasoning, "executor not yet wired to model client");
            }
            other => panic!("expected Defer, got {other:?}"),
        }
    }
}
