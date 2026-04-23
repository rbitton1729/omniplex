//! Inter-agent Unix socket bus.
//!
//! Every agent binds a Unix socket at a well-known path:
//!   <socket_dir>/<agent_name>.sock
//!
//! Other agents (or external processes) discover whom to talk to by listing
//! the directory.  Messages are newline-delimited JSON of [`PeerMessage`];
//! replies are newline-delimited JSON of [`PeerReply`].
//!
//! Phase 1 scope:
//!   - One-shot request/response over a connected UnixStream.
//!   - No multiplexing; a new connection per message.
//!   - No encryption over the socket (it is local Unix).
//!   - Bus is responsible for transport only; deliberation and budgeting stay
//!     in the executor.
//!
//! Phase 2 will add:
//!   - Streaming replies (deliberation may take minutes; caller should be able
//!     to read partial thinking).
//!   - Unix credential passing (SO_PEERCRED) so receivers can authenticate
//!     the sending process without relying on the JSON `from` field.
//!   - Broadcast/multicast (fan-out to all agents matching a glob).

use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufStream};
use tokio::net::UnixStream;

use crate::types::{AgentId, BudgetSnapshot, Deliberation, Task, Timestamp};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum BusError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("agent not available: {0}")]
    AgentUnavailable(String),
    #[error("timeout waiting for reply from {0}")]
    Timeout(String),
}

// ---------------------------------------------------------------------------
// Wire protocol
// ---------------------------------------------------------------------------

/// Origin of a peer message.  The `from` field is advisory; Phase 2 will
/// harden this with SO_PEERCRED.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PeerOrigin {
    /// Message sent by a named Omniplex agent over the socket bus.
    Agent { name: String },
    /// Message injected by an external tool (e.g. `nc -U`, a webhook
    /// forwarder, or a systemd pipe).
    External { label: String },
    /// Message originating from the runtime itself (parent daemon,
    /// signal handlers, etc.).
    Runtime,
}

/// The envelope every agent receives on its Unix socket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PeerMessage {
    // ------------------------------------------------------------------
    // Human or tool talking to an agent (Phase 1)
    // ------------------------------------------------------------------
    /// A plain text message.  The agent deliberates and replies.
    Message {
        from: PeerOrigin,
        text: String,
        /// When this message was generated (useful for agents that care about
        /// wall-clock silence, e.g. Barnaby).
        sent_at: Timestamp,
    },

    // ------------------------------------------------------------------
    // Agent-to-agent delegation (Phase 1)
    // ------------------------------------------------------------------
    /// Another agent or the runtime wants this agent to perform a task.
    /// The budget snapshot is what the *caller* is willing to give the
    /// callee; the callee may reserve less and return the rest on drop.
    Delegate {
        from: PeerOrigin,
        task: Task,
        budget: BudgetSnapshot,
        /// A short label for tracing ("ci-watcher pipeline 7 investigation").
        label: String,
    },

    // ------------------------------------------------------------------
    // Event injection (Phase 1 for CI, Phase 2 for UserQuiet/Memory)
    // ------------------------------------------------------------------
    /// An external event matched a watch.  The agent deliberates whether
    /// to Act / Defer / Abandon based on its current state.
    Event {
        from: PeerOrigin,
        payload: EventPayload,
        /// ISO-8601 string; parsed to Timestamp by receiver.
        occurred_at: Timestamp,
    },

    // ------------------------------------------------------------------
    // Control (Phase 1)
    // ------------------------------------------------------------------
    /// Liveness probe.  Receiver replies with `PeerReply::Ack`.
    Ping,
    /// Ask the agent for a structured status dump without triggering a
    /// deliberation (cheap, read-only).
    Status,
}

/// Events that can be injected into an agent from the outside world.
/// These map 1:1 to future `Watch` variants, but arrive "out of band"
/// when something else (inotify, webhook, systemd timer) detects them
/// and forwards them over the socket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EventPayload {
    /// GitLab CI pipeline finished.
    CiFinished {
        project: String,
        branch: String,
        status: CiStatus,
        pipeline_url: String,
    },
    /// File changed on disk (inotify).
    FileChanged { path: String, event: FileEventKind },
    /// POSIX signal delivered to the process (phase 2).
    Signal { signal: String },
    /// Generic catch-all for Phase 1 until we have strongly-typed variants.
    Raw { json: serde_json::Value },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CiStatus {
    Success,
    Failed,
    Running,
    Cancelled,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileEventKind {
    Modified,
    Created,
    Deleted,
    Renamed,
}

// ---------------------------------------------------------------------------
// Reply envelope
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PeerReply {
    /// The agent's deliberation result (Act / Defer / Abandon).
    Deliberation(Deliberation),
    /// Liveness / status acknowledgement.
    Ack,
    /// Structured read-only status.  Phase 1 returns budget snapshot + agent id.
    Status {
        agent: AgentId,
        budget: BudgetSnapshot,
        /// Wall-clock uptime of the process.
        uptime_secs: u64,
    },
    /// Something went wrong inside the agent trying to handle the message.
    Error { reason: String },
}

// ---------------------------------------------------------------------------
// Client-side helpers (anyone talking *to* an agent)
// ---------------------------------------------------------------------------

/// Resolve a well-known socket path for an agent by name.
pub fn socket_path_for(socket_dir: &Path, name: &str) -> PathBuf {
    socket_dir.join(format!("{name}.sock"))
}

/// List agent names currently advertising sockets in `socket_dir`.
pub fn list_agents(socket_dir: &Path) -> io::Result<Vec<String>> {
    let mut out = vec![];
    if !socket_dir.exists() {
        return Ok(out);
    }
    for entry in std::fs::read_dir(socket_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(s) = name.to_str() else { continue };
        if s.ends_with(".sock") {
            out.push(s.trim_end_matches(".sock").to_string());
        }
    }
    Ok(out)
}

/// Send a `PeerMessage` to an agent and await its `PeerReply`.
/// One connection, one request, one reply, connection closed.
pub async fn send_one(
    socket_dir: &Path,
    agent: &str,
    msg: &PeerMessage,
) -> Result<PeerReply, BusError> {
    let path = socket_path_for(socket_dir, agent);
    if !path.exists() {
        return Err(BusError::AgentUnavailable(agent.to_string()));
    }
    let stream = UnixStream::connect(&path).await?;
    let mut framed = BufStream::new(stream);

    let json = serde_json::to_vec(msg)?;
    framed.write_all(&json).await?;
    framed.write_all(b"\n").await?;
    framed.flush().await?;

    let mut line = String::new();
    let n = framed.read_line(&mut line).await?;
    if n == 0 {
        return Err(BusError::AgentUnavailable(agent.to_string()));
    }

    let reply: PeerReply = serde_json::from_str(&line)?;
    Ok(reply)
}

// ---------------------------------------------------------------------------
// Server-side helpers (agent listening on its own socket)
// ---------------------------------------------------------------------------

/// Accept a single message from a connected `UnixStream`, parse it, and
/// return the `PeerMessage`.
pub async fn recv_one(stream: UnixStream) -> Result<PeerMessage, BusError> {
    let mut framed = BufStream::new(stream);
    let mut line = String::new();
    let n = framed.read_line(&mut line).await?;
    if n == 0 {
        return Err(BusError::Io(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "peer closed before sending message",
        )));
    }
    let msg: PeerMessage = serde_json::from_str(&line)?;
    Ok(msg)
}

/// Write a `PeerReply` to a connected `UnixStream` and flush.
pub async fn reply_one(stream: &mut UnixStream, reply: &PeerReply) -> Result<(), BusError> {
    let json = serde_json::to_vec(reply)?;
    stream.write_all(&json).await?;
    stream.write_all(b"\n").await?;
    stream.flush().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::UnixListener;
    #[tokio::test]
    async fn roundtrip_message_over_socket() {
        let tmp = std::env::temp_dir().join(format!("omniplex-test-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let listener = UnixListener::bind(&tmp).unwrap();

        let srv = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let msg = recv_one(stream).await.unwrap();
            let (mut stream, _) = listener.accept().await.unwrap();
            let reply = match msg {
                PeerMessage::Ping => PeerReply::Ack,
                PeerMessage::Message { .. } => {
                    PeerReply::Deliberation(crate::types::Deliberation::Defer {
                        until: crate::types::DeferCondition::Never,
                        reasoning: "testing".into(),
                    })
                }
                _ => PeerReply::Error {
                    reason: "unexpected".into(),
                },
            };
            reply_one(&mut stream, &reply).await.unwrap();
        });

        let msg = PeerMessage::Message {
            from: PeerOrigin::External {
                label: "test".into(),
            },
            text: "hello".into(),
            sent_at: chrono::Utc::now(),
        };
        // TODO: fixup to connect to the same socket after accept in server
        let reply = send_one(tmp.parent().unwrap(), "omniplex-test", &msg).await;
        // This test is skeleton-only because send_one and recv_one currently
        // model one-shot request/response on the *same* connection.  The real
        // harness pairs accept() + recv_one + reply_one on a single stream.
        // Leave this as a placeholder until the executor loop lands.
        let _ = (srv, reply);
    }
}
