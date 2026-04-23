//! Memory service: Lantern-backed activity recording and recall.
//!
//! Agents write deliberation turns as JSONL-shaped activity records. Each
//! turn becomes one Lantern source containing N chunks (one per JSONL line,
//! via Lantern's built-in jsonl extractor). Agent identity rides on the
//! source URI: `agent://<name>/goal/<goal_id>/turn/<turn_id>`.
//!
//! Lantern is sync (rusqlite). This module wraps it in an actor that owns
//! the `Store` on a dedicated OS thread; the async trait methods send ops
//! over mpsc and await oneshot replies.
//!
//! Phase 1 scope:
//!   - Keyword (BM25) search only. Semantic/Hybrid variants return
//!     `MemoryError::NotImplemented` until the embed worker lands.
//!   - No scope enforcement on reads. URIs are structured so that adding
//!     `WHERE uri LIKE 'agent://<allowed>/%'` in Phase 3 is a one-liner.
//!   - Immediate write-per-turn. No Periodic buffering yet; SQLite WAL
//!     mode handles our write rate. Batching lands with commit policies
//!     in Phase 3.

use std::path::Path;
use std::thread;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::types::{AgentId, Timestamp};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("lantern: {0}")]
    Lantern(String),
    #[error("memory actor shut down")]
    Shutdown,
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("not implemented in phase 1: {0}")]
    NotImplemented(&'static str),
}

/// One entry inside a deliberation turn. Becomes a single JSONL line, which
/// Lantern's jsonl extractor chunks and populates with `role`, `session_id`,
/// `turn_id`, `tool_name`, and `timestamp_unix` for later retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnEntry {
    /// "deliberation" | "task_intent" | "task_result" | "observation" | "note"
    pub role: String,
    pub content: String,
    pub timestamp: Timestamp,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

/// A complete deliberation turn — the unit a harness commits in one shot.
/// Persisted as one Lantern source at `agent://<name>/goal/<goal>/turn/<turn>`.
#[derive(Debug, Clone)]
pub struct ActivityTurn {
    pub goal_id: Uuid,
    pub turn_id: Uuid,
    pub entries: Vec<TurnEntry>,
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub uri: String,
    pub ordinal: usize,
    pub text: String,
    pub snippet: String,
    pub score: f64,
    pub role: Option<String>,
    pub session_id: Option<String>,
    pub turn_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SearchMode {
    #[default]
    Keyword,
    Semantic,
    Hybrid,
}

#[derive(Debug, Clone)]
pub struct SearchOpts {
    pub mode: SearchMode,
    pub limit: usize,
}

impl Default for SearchOpts {
    fn default() -> Self {
        Self {
            mode: SearchMode::Keyword,
            limit: 10,
        }
    }
}

// ---------------------------------------------------------------------------
// URI scheme
// ---------------------------------------------------------------------------

pub fn turn_uri(agent: &AgentId, goal_id: Uuid, turn_id: Uuid) -> String {
    format!("agent://{}/goal/{}/turn/{}", agent.name, goal_id, turn_id)
}

pub fn agent_scope_prefix(agent: &AgentId) -> String {
    format!("agent://{}/", agent.name)
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait MemoryService: Send + Sync {
    /// Persist a deliberation turn as a single Lantern source.
    async fn record_turn(&self, agent: &AgentId, turn: ActivityTurn) -> Result<(), MemoryError>;

    /// Active recall: user-style query against an agent's activity.
    async fn search(
        &self,
        agent: &AgentId,
        query: &str,
        opts: SearchOpts,
    ) -> Result<Vec<Hit>, MemoryError>;

    /// Passive recall: surface context relevant to `context` for injection
    /// into the next deliberation. Distinct from `search` only in intended
    /// use; Phase 1 both delegate to BM25 under the hood.
    async fn recall(
        &self,
        agent: &AgentId,
        context: &str,
        opts: SearchOpts,
    ) -> Result<Vec<Hit>, MemoryError>;
}

// ---------------------------------------------------------------------------
// Lantern-backed implementation
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct LanternMemory {
    tx: mpsc::Sender<Op>,
}

enum Op {
    Record {
        uri: String,
        jsonl: Vec<u8>,
        reply: oneshot::Sender<Result<(), MemoryError>>,
    },
    Search {
        prefix: String,
        query: String,
        opts: SearchOpts,
        reply: oneshot::Sender<Result<Vec<Hit>, MemoryError>>,
    },
}

impl LanternMemory {
    /// Open or initialize a Lantern store at `root` and spawn the actor.
    /// Root is the `.lantern/` directory, not the sqlite file itself.
    pub fn spawn(root: &Path) -> Result<Self, MemoryError> {
        let store = lantern::store::Store::initialize(root)
            .map_err(|e| MemoryError::Lantern(e.to_string()))?;

        let (tx, rx) = mpsc::channel::<Op>(64);
        thread::Builder::new()
            .name("lantern-memory".into())
            .spawn(move || actor_loop(store, rx))
            .map_err(|e| MemoryError::Lantern(format!("spawn thread: {e}")))?;

        Ok(Self { tx })
    }

    async fn send(&self, op: Op) -> Result<(), MemoryError> {
        self.tx.send(op).await.map_err(|_| MemoryError::Shutdown)
    }
}

#[async_trait]
impl MemoryService for LanternMemory {
    async fn record_turn(&self, agent: &AgentId, turn: ActivityTurn) -> Result<(), MemoryError> {
        let uri = turn_uri(agent, turn.goal_id, turn.turn_id);
        let jsonl = encode_turn_jsonl(&turn)?;

        let (reply, rx) = oneshot::channel();
        self.send(Op::Record { uri, jsonl, reply }).await?;
        rx.await.map_err(|_| MemoryError::Shutdown)?
    }

    async fn search(
        &self,
        agent: &AgentId,
        query: &str,
        opts: SearchOpts,
    ) -> Result<Vec<Hit>, MemoryError> {
        let (reply, rx) = oneshot::channel();
        self.send(Op::Search {
            prefix: agent_scope_prefix(agent),
            query: query.to_string(),
            opts,
            reply,
        })
        .await?;
        rx.await.map_err(|_| MemoryError::Shutdown)?
    }

    async fn recall(
        &self,
        agent: &AgentId,
        context: &str,
        opts: SearchOpts,
    ) -> Result<Vec<Hit>, MemoryError> {
        // Phase 1: same path as search.
        self.search(agent, context, opts).await
    }
}

// ---------------------------------------------------------------------------
// Actor loop + Lantern bridging
// ---------------------------------------------------------------------------

fn actor_loop(mut store: lantern::store::Store, mut rx: mpsc::Receiver<Op>) {
    while let Some(op) = rx.blocking_recv() {
        match op {
            Op::Record { uri, jsonl, reply } => {
                let _ = reply.send(do_record(&mut store, &uri, &jsonl));
            }
            Op::Search {
                prefix,
                query,
                opts,
                reply,
            } => {
                let _ = reply.send(do_search(&store, &prefix, &query, opts));
            }
        }
    }
}

fn do_record(
    store: &mut lantern::store::Store,
    uri: &str,
    jsonl: &[u8],
) -> Result<(), MemoryError> {
    let report = lantern::ingest::ingest_stdin(store, uri, Some("application/jsonl"), jsonl)
        .map_err(|e| MemoryError::Lantern(e.to_string()))?;
    // `ingest_stdin` reports either ingested, skipped-unchanged (same URI
    // and same content), or skipped-error. For us, re-recording a turn URI
    // with identical bytes is a no-op and fine; an error skip is fatal.
    if let Some(skip) = report.skipped.first() {
        // Lantern's SkippedSource has a category (`skipped_reason`: "unchanged"
        // | "error" | ...) and a detail (`reason`). Re-ingesting identical
        // content is expected and fine; anything else is fatal.
        if skip.skipped_reason.as_deref() != Some("unchanged") {
            return Err(MemoryError::Lantern(format!(
                "ingest skipped {}: {}",
                skip.uri, skip.reason
            )));
        }
    }
    Ok(())
}

fn do_search(
    store: &lantern::store::Store,
    prefix: &str,
    query: &str,
    opts: SearchOpts,
) -> Result<Vec<Hit>, MemoryError> {
    match opts.mode {
        SearchMode::Keyword => {}
        SearchMode::Semantic => {
            return Err(MemoryError::NotImplemented(
                "semantic search (Ollama embeddings)",
            ));
        }
        SearchMode::Hybrid => {
            return Err(MemoryError::NotImplemented(
                "hybrid search (requires embeddings)",
            ));
        }
    }

    // NOTE: Phase 1 intentionally does NOT enforce agent scope — see module
    // docs. The URI prefix is passed through so the same call site will work
    // unchanged when scope enforcement lands.
    let _ = prefix;

    let hits = lantern::search::search(
        store,
        query,
        lantern::search::SearchOptions {
            limit: opts.limit,
            kind: None,
            path_contains: None,
        },
    )
    .map_err(|e| MemoryError::Lantern(e.to_string()))?;

    Ok(hits.into_iter().map(map_hit).collect())
}

fn map_hit(h: lantern::search::SearchHit) -> Hit {
    Hit {
        uri: h.uri,
        ordinal: h.ordinal.max(0) as usize,
        text: h.text,
        snippet: h.snippet,
        score: h.score,
        role: h.role,
        session_id: h.session_id,
        turn_id: h.turn_id,
    }
}

fn encode_turn_jsonl(turn: &ActivityTurn) -> Result<Vec<u8>, MemoryError> {
    if turn.entries.is_empty() {
        return Err(MemoryError::Invalid("turn has no entries".into()));
    }
    let mut out = Vec::with_capacity(turn.entries.len() * 128);
    for entry in &turn.entries {
        let line = serde_json::json!({
            "role": entry.role,
            "session_id": turn.goal_id.to_string(),
            "turn_id": turn.turn_id.to_string(),
            "tool_name": entry.tool_name,
            "timestamp_unix": entry.timestamp.timestamp(),
            "content": entry.content,
        });
        serde_json::to_writer(&mut out, &line)
            .map_err(|e| MemoryError::Invalid(format!("jsonl encode: {e}")))?;
        out.push(b'\n');
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use tempfile::TempDir;

    fn entry(role: &str, content: &str) -> TurnEntry {
        TurnEntry {
            role: role.into(),
            content: content.into(),
            timestamp: Utc::now(),
            tool_name: None,
        }
    }

    fn turn(entries: Vec<TurnEntry>) -> ActivityTurn {
        ActivityTurn {
            goal_id: Uuid::new_v4(),
            turn_id: Uuid::new_v4(),
            entries,
        }
    }

    #[tokio::test]
    async fn record_and_search_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let mem = LanternMemory::spawn(tmp.path()).unwrap();
        let barnaby = AgentId::new("barnaby");

        let t = turn(vec![
            entry(
                "deliberation",
                "Three CI failures on pipeline seven in the last hour.",
            ),
            entry("task_intent", "Escalate to oncall for pipeline seven."),
        ]);
        mem.record_turn(&barnaby, t).await.unwrap();

        // BM25 search on a single token the jsonl extractor preserves inline.
        let hits = mem
            .search(&barnaby, "escalate", SearchOpts::default())
            .await
            .unwrap();
        assert!(!hits.is_empty(), "expected at least one hit");
        assert!(
            hits.iter()
                .any(|h| h.text.to_lowercase().contains("escalate"))
        );
        assert!(hits[0].uri.starts_with("agent://barnaby/goal/"));
        // jsonl extractor prefixes with role — confirm metadata round-tripped.
        assert!(
            hits.iter()
                .any(|h| h.role.as_deref() == Some("task_intent"))
        );
    }

    #[tokio::test]
    async fn empty_turn_is_rejected() {
        let tmp = TempDir::new().unwrap();
        let mem = LanternMemory::spawn(tmp.path()).unwrap();
        let barnaby = AgentId::new("barnaby");
        let err = mem.record_turn(&barnaby, turn(vec![])).await.unwrap_err();
        assert!(matches!(err, MemoryError::Invalid(_)));
    }

    #[tokio::test]
    async fn semantic_mode_reports_not_implemented() {
        let tmp = TempDir::new().unwrap();
        let mem = LanternMemory::spawn(tmp.path()).unwrap();
        let barnaby = AgentId::new("barnaby");
        mem.record_turn(&barnaby, turn(vec![entry("note", "hello")]))
            .await
            .unwrap();

        let opts = SearchOpts {
            mode: SearchMode::Semantic,
            limit: 10,
        };
        let err = mem.search(&barnaby, "anything", opts).await.unwrap_err();
        assert!(matches!(err, MemoryError::NotImplemented(_)));
    }

    #[tokio::test]
    async fn re_record_same_turn_is_idempotent() {
        let tmp = TempDir::new().unwrap();
        let mem = LanternMemory::spawn(tmp.path()).unwrap();
        let barnaby = AgentId::new("barnaby");

        let t = turn(vec![entry("deliberation", "once")]);
        mem.record_turn(&barnaby, t.clone()).await.unwrap();
        // Same turn_id + same content → Lantern reports "unchanged", we treat as OK.
        mem.record_turn(&barnaby, t).await.unwrap();
    }

    #[test]
    fn turn_uri_is_stable_and_scoped() {
        let a = AgentId::new("barnaby");
        let g = Uuid::nil();
        let t = Uuid::nil();
        let uri = turn_uri(&a, g, t);
        assert!(uri.starts_with("agent://barnaby/"));
        assert!(uri.contains("/goal/"));
        assert!(uri.contains("/turn/"));
    }
}
