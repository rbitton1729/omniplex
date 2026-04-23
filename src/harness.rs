//! Agent harness: per-agent Unix socket accept loop.
//!
//! Each harness owns one `UnixListener` at `<socket_dir>/<name>.sock` and
//! services one `PeerMessage` per accepted connection. The reply is written
//! on the same stream and the connection is closed — matching the one-shot
//! request/response contract described in `bus.rs`.
//!
//! Phase 0 routing is intentionally trivial:
//!   - `Ping`   → `Ack`
//!   - `Status` → `Status { agent, budget-from-policy, uptime }`
//!   - anything that would normally deliberate (`Message`, `Delegate`,
//!     `Event`) → `Deliberation(Defer { Never, "executor not wired" })`
//!
//! Phase 1 will replace the deliberation arm with a real executor call;
//! the socket/accept machinery here does not need to change when that lands.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufStream};
use tokio::net::{UnixListener, UnixStream};

use crate::bus::{PeerMessage, PeerReply};
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
}

impl AgentHarness {
    /// Bind `socket_path` and prepare the accept loop. Caller is responsible
    /// for ensuring the parent directory exists and the path is free (stale
    /// sockets from a previous run must be unlinked first — `UnixListener`
    /// will otherwise fail with `AddrInUse`).
    pub fn bind(config: AgentConfig, socket_path: PathBuf) -> Result<Self, HarnessError> {
        let listener = UnixListener::bind(&socket_path)?;
        Ok(Self {
            config: Arc::new(config),
            listener,
            socket_path,
            started: Instant::now(),
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
            let started = self.started;
            let name = agent_name.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_connection(cfg, started, stream).await {
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
        Ok(msg) => route(&cfg, started, msg),
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

fn route(cfg: &AgentConfig, started: Instant, msg: PeerMessage) -> PeerReply {
    match msg {
        PeerMessage::Ping => PeerReply::Ack,
        PeerMessage::Status => PeerReply::Status {
            agent: cfg.id.clone(),
            budget: BudgetSnapshot {
                remaining: cfg.default_budget.allocation(),
                depth: 0,
                max_depth: cfg.default_budget.max_depth,
            },
            uptime_secs: started.elapsed().as_secs(),
        },
        PeerMessage::Message { .. } | PeerMessage::Delegate { .. } | PeerMessage::Event { .. } => {
            PeerReply::Deliberation(Deliberation::Defer {
                until: DeferCondition::Never,
                reasoning: format!(
                    "{}: deliberation executor not yet wired (phase 0 harness)",
                    cfg.id.name
                ),
            })
        }
    }
}
