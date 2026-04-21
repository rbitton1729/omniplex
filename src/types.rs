//! Omniplex core type definitions — Phase 0 draft.
//!
//! These are the shapes that appear in every function signature and every
//! trace record. They need to be right before implementation starts, because
//! changing them later is a whole-repo edit.
//!
//! Scope: types only. No executor, harness, memory, or I/O.
//!
//! Design decisions settled in the pre-draft conversation:
//!   - Budget uses Arc<Mutex<BudgetCell>>; Reservation holds the Arc and
//!     releases unspent tokens on Drop. Crosses .await freely.
//!   - AgentId is { name }. UUID is derived once at registry-insert via
//!     UUIDv5 and stored; rename uses the alias table, never re-derives.
//!   - SideEffectClass is declared on Capability, not Task.
//!   - Deliberation Phase 1 subset: Act / Defer / Abandon. NeedMore lands
//!     in Phase 2 with deliberative watches.
//!   - Model pricing / context window live in a ModelCatalog (not drafted
//!     here); AgentConfig references models by provider + name only.
//!
//! Open (flagged in review message, not this file):
//!   - Lantern integration shape: embedded crate vs MCP client vs CLI.
//!   - AGENT_NAMESPACE UUID value (placeholder below).
//!   - EventPattern / EventFilter / UserId proper shapes — land alongside
//!     the first watch implementation in Phase 1.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub type Timestamp = chrono::DateTime<chrono::Utc>;

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

/// Tokens + calls, as either a remaining balance or an amount to subtract.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Allocation {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub calls: u32,
}

impl Allocation {
    pub const fn zero() -> Self {
        Self { input_tokens: 0, output_tokens: 0, calls: 0 }
    }
}

/// Observed usage from a single model call. Charged against a Reservation.
#[derive(Debug, Clone, Copy)]
pub struct ModelUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Error)]
pub enum BudgetExhausted {
    #[error("input tokens exhausted")]
    InputTokens,
    #[error("output tokens exhausted")]
    OutputTokens,
    #[error("call count exhausted")]
    Calls,
    #[error("max delegation depth reached")]
    Depth,
}

#[derive(Debug)]
struct BudgetCell {
    remaining: Allocation,
    max_depth: u8,
    current_depth: u8,
}

/// Cheap to clone (Arc-backed). Send + Sync. Safe to hold across .await.
/// Clones share state: a reservation on one clone charges the same cell as
/// any other clone. For independent children, use [`Budget::delegate_concurrent`].
#[derive(Clone, Debug)]
pub struct Budget {
    cell: Arc<Mutex<BudgetCell>>,
    pub origin: AgentId,
    pub task_id: Uuid,
}

impl Budget {
    pub fn new(origin: AgentId, task_id: Uuid, allocation: Allocation, max_depth: u8) -> Self {
        Self {
            cell: Arc::new(Mutex::new(BudgetCell {
                remaining: allocation,
                max_depth,
                current_depth: 0,
            })),
            origin,
            task_id,
        }
    }

    /// Reserve up to `requested`. The returned Reservation releases any
    /// unspent portion back to this budget on Drop. Charge actual spend
    /// via [`Reservation::charge`] after each model call.
    pub fn reserve(&self, requested: Allocation) -> Result<Reservation, BudgetExhausted> {
        let mut cell = self.cell.lock().expect("budget mutex poisoned");
        check_and_subtract(&mut cell.remaining, &requested)?;
        Ok(Reservation {
            cell: self.cell.clone(),
            reserved: requested,
            spent: Allocation::zero(),
        })
    }

    /// Split off an independent child budget for concurrent delegation.
    /// The parent permanently loses the allocation — no Drop-reclaim.
    /// Use when a delegate runs in parallel with the parent or outlives it.
    pub fn delegate_concurrent(&self, allocation: Allocation) -> Result<Budget, BudgetExhausted> {
        let mut cell = self.cell.lock().expect("budget mutex poisoned");
        if cell.current_depth >= cell.max_depth {
            return Err(BudgetExhausted::Depth);
        }
        check_and_subtract(&mut cell.remaining, &allocation)?;
        let child_depth = cell.current_depth + 1;
        let max_depth = cell.max_depth;
        Ok(Budget {
            cell: Arc::new(Mutex::new(BudgetCell {
                remaining: allocation,
                max_depth,
                current_depth: child_depth,
            })),
            origin: self.origin.clone(),
            task_id: self.task_id,
        })
    }

    pub fn snapshot(&self) -> BudgetSnapshot {
        let cell = self.cell.lock().expect("budget mutex poisoned");
        BudgetSnapshot {
            remaining: cell.remaining,
            depth: cell.current_depth,
            max_depth: cell.max_depth,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BudgetSnapshot {
    pub remaining: Allocation,
    pub depth: u8,
    pub max_depth: u8,
}

/// Scoped reservation. Drop releases `reserved - spent` back to the parent.
#[must_use = "reservations release on Drop; don't discard unless intended"]
pub struct Reservation {
    cell: Arc<Mutex<BudgetCell>>,
    reserved: Allocation,
    spent: Allocation,
}

impl Reservation {
    /// Charge observed usage. Errors if charge would exceed the reservation;
    /// in that case the reservation is unchanged and the caller should drop it.
    pub fn charge(&mut self, usage: ModelUsage) -> Result<(), BudgetExhausted> {
        let input = self.spent.input_tokens.saturating_add(usage.input_tokens);
        let output = self.spent.output_tokens.saturating_add(usage.output_tokens);
        let calls = self.spent.calls.saturating_add(1);

        if input > self.reserved.input_tokens {
            return Err(BudgetExhausted::InputTokens);
        }
        if output > self.reserved.output_tokens {
            return Err(BudgetExhausted::OutputTokens);
        }
        if calls > self.reserved.calls {
            return Err(BudgetExhausted::Calls);
        }
        self.spent = Allocation { input_tokens: input, output_tokens: output, calls };
        Ok(())
    }

    pub fn spent(&self) -> Allocation {
        self.spent
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let unused = Allocation {
            input_tokens: self.reserved.input_tokens.saturating_sub(self.spent.input_tokens),
            output_tokens: self.reserved.output_tokens.saturating_sub(self.spent.output_tokens),
            calls: self.reserved.calls.saturating_sub(self.spent.calls),
        };
        if let Ok(mut cell) = self.cell.lock() {
            cell.remaining.input_tokens =
                cell.remaining.input_tokens.saturating_add(unused.input_tokens);
            cell.remaining.output_tokens =
                cell.remaining.output_tokens.saturating_add(unused.output_tokens);
            cell.remaining.calls = cell.remaining.calls.saturating_add(unused.calls);
        }
    }
}

fn check_and_subtract(
    remaining: &mut Allocation,
    amount: &Allocation,
) -> Result<(), BudgetExhausted> {
    remaining.input_tokens = remaining
        .input_tokens
        .checked_sub(amount.input_tokens)
        .ok_or(BudgetExhausted::InputTokens)?;
    remaining.output_tokens = remaining
        .output_tokens
        .checked_sub(amount.output_tokens)
        .ok_or(BudgetExhausted::OutputTokens)?;
    remaining.calls = remaining
        .calls
        .checked_sub(amount.calls)
        .ok_or(BudgetExhausted::Calls)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Agent identity
// ---------------------------------------------------------------------------

// Placeholder. Pick a real UUID once and pin it forever: the derivation of
// every agent UUID depends on this namespace being stable.
pub const AGENT_NAMESPACE: Uuid = Uuid::from_u128(0x6f6d6e69_706c_7800_0000_000000000001);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AgentId {
    pub name: String,
}

impl AgentId {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// One-shot UUIDv5 derivation used when first inserting an agent into
    /// the registry. After insert, lookups go through the registry (with
    /// alias resolution); callers MUST NOT re-derive on every access or
    /// renames will orphan Lantern chunks.
    pub fn bootstrap_uuid(&self) -> Uuid {
        Uuid::new_v5(&AGENT_NAMESPACE, self.name.as_bytes())
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("agent not found: {0}")]
    NotFound(String),
    #[error("agent already registered: {0}")]
    AlreadyExists(String),
    #[error("alias would collide with existing agent: {0}")]
    AliasCollision(String),
}

/// Name → AgentConfig + stable UUID. Rename updates an alias table so that
/// old names continue to resolve to the same UUID (Lantern chunks keyed by
/// UUID stay valid).
pub trait Registry: Send + Sync {
    fn get(&self, name: &str) -> Result<Arc<AgentConfig>, RegistryError>;
    fn uuid_of(&self, name: &str) -> Result<Uuid, RegistryError>;
    fn insert(&self, config: AgentConfig) -> Result<Uuid, RegistryError>;
    fn rename(&self, from: &str, to: &str) -> Result<(), RegistryError>;
    fn list(&self) -> Vec<String>;
}

// ---------------------------------------------------------------------------
// AgentConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub id: AgentId,
    pub model: ModelRef,
    pub system_prompt: String,
    #[serde(default)]
    pub memory_scope: MemoryScope,
    #[serde(default)]
    pub allowed_capabilities: Vec<CapabilityId>,
    pub default_budget: BudgetPolicy,
    pub commit_policy: CommitPolicy,
}

/// Reference into the model catalog. Pricing and max-context live in the
/// catalog, keyed by (provider, model), so provider pricing changes don't
/// require editing every agent TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef {
    pub provider: ModelProvider,
    pub model: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelProvider {
    Anthropic,
    Ollama,
    OpenAiCompatible,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MemoryScope {
    #[default]
    Private,
    Shared { with: Vec<AgentId> },
    Global,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetPolicy {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub calls: u32,
    pub max_depth: u8,
}

impl BudgetPolicy {
    pub fn allocation(&self) -> Allocation {
        Allocation {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            calls: self.calls,
        }
    }

    pub fn into_budget(&self, origin: AgentId, task_id: Uuid) -> Budget {
        Budget::new(origin, task_id, self.allocation(), self.max_depth)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CommitPolicy {
    Manual,
    OnCheckpoint,
    Periodic {
        #[serde(with = "humantime_serde")]
        interval: Duration,
    },
    OnShutdown,
}

// ---------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CapabilityId(pub String);

/// Correctness guarantees for a capability's side effects. Declared on the
/// capability, inherited by any Task that invokes it — agents can't
/// downgrade the guarantees of a tool they call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SideEffectClass {
    /// No observable side effect. Retry / discard freely.
    Pure,
    /// External system dedupes on idempotency_key. Commit-after-effect OK.
    Idempotent,
    /// Must not duplicate. Harness commits intent BEFORE effect; on restart,
    /// replay relies on the external system honoring the idempotency key.
    ExactlyOnce,
    /// Opt-in. Commit-after-effect; restart may duplicate.
    AtLeastOnce,
}

#[derive(Debug, Clone)]
pub struct Capability {
    pub id: CapabilityId,
    pub description: String,
    pub side_effect: SideEffectClass,
    /// Per-capability model override for escalate (frontier reasoning) or
    /// de-escalate (local classification). None means use the agent's default.
    pub model_override: Option<ModelRef>,
}

// ---------------------------------------------------------------------------
// Goals and watches
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Goal {
    pub id: Uuid,
    pub agent: AgentId,
    pub watch: Watch,
    pub created: Timestamp,
    pub last_check: Option<Timestamp>,
}

/// Phase 1 ships `Pattern` and `Once`. The others are sketched here because
/// their existence affects the `Watch` enum's shape; implementations land
/// with their phases (UserQuiet/Memory in Phase 2, Ci in Phase 1).
#[derive(Debug, Clone)]
pub enum Watch {
    Pattern { pattern: EventPattern, threshold: usize, window: Duration },
    UserQuiet { user: UserId, duration: Duration },
    Ci { project: String, expected: CiStatus },
    Memory { query: String },
    Once { payload: serde_json::Value },
}

// Placeholder shapes. The proper types land with the first watch impl;
// they live here to keep `Watch` compiling.
#[derive(Debug, Clone)]
pub struct EventPattern(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UserId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiStatus {
    Success,
    Failed,
    Running,
}

// ---------------------------------------------------------------------------
// Deliberation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Task {
    pub capability: CapabilityId,
    pub args: serde_json::Value,
    /// Deterministic key: hash(goal_id || deliberation_id || capability || args_canonical).
    /// External systems classified `Idempotent` or `ExactlyOnce` dedupe on this.
    pub idempotency_key: String,
}

/// Phase 1 subset. `NeedMore` (model requests more context) lands in Phase 2
/// with deliberative watches. Adding a variant is a breaking change for the
/// executor's dispatch; do it once, intentionally.
#[derive(Debug, Clone)]
pub enum Deliberation {
    Act {
        task: Task,
        reasoning: String,
        next_check: DeferCondition,
    },
    Defer {
        until: DeferCondition,
        reasoning: String,
    },
    Abandon {
        reasoning: String,
    },
}

#[derive(Debug, Clone)]
pub enum DeferCondition {
    Time(Timestamp),
    Event(EventFilter),
    Either(Box<DeferCondition>, Box<DeferCondition>),
    /// "Don't poll; only wake on explicit events." Mechanical watches
    /// whose only trigger is an event stream use this.
    Never,
}

#[derive(Debug, Clone)]
pub struct EventFilter(pub String);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn origin() -> AgentId {
        AgentId::new("barnaby")
    }

    fn alloc(input: u64, output: u64, calls: u32) -> Allocation {
        Allocation { input_tokens: input, output_tokens: output, calls }
    }

    #[test]
    fn reservation_returns_unused_tokens_on_drop() {
        let budget = Budget::new(origin(), Uuid::new_v4(), alloc(1000, 1000, 10), 3);

        {
            let mut r = budget.reserve(alloc(500, 500, 5)).unwrap();
            r.charge(ModelUsage { input_tokens: 100, output_tokens: 50 }).unwrap();
            // Drop happens here; unspent 400/450/4 returns.
        }

        let snap = budget.snapshot();
        assert_eq!(snap.remaining, alloc(900, 950, 9));
    }

    #[test]
    fn charge_beyond_reservation_errors_and_leaves_state() {
        let budget = Budget::new(origin(), Uuid::new_v4(), alloc(1000, 1000, 10), 3);
        let mut r = budget.reserve(alloc(100, 100, 2)).unwrap();
        r.charge(ModelUsage { input_tokens: 80, output_tokens: 80 }).unwrap();

        let err = r
            .charge(ModelUsage { input_tokens: 50, output_tokens: 10 })
            .unwrap_err();
        assert!(matches!(err, BudgetExhausted::InputTokens));
        // Prior charge stands.
        assert_eq!(r.spent(), alloc(80, 80, 1));
    }

    #[test]
    fn delegate_concurrent_is_split_no_reclaim() {
        let parent = Budget::new(origin(), Uuid::new_v4(), alloc(1000, 1000, 10), 3);
        let child = parent.delegate_concurrent(alloc(300, 300, 3)).unwrap();

        // Child unused on drop — parent does NOT get it back.
        drop(child);

        assert_eq!(parent.snapshot().remaining, alloc(700, 700, 7));
    }

    #[test]
    fn delegate_concurrent_enforces_depth() {
        let budget = Budget::new(origin(), Uuid::new_v4(), alloc(1000, 1000, 10), 1);
        let child = budget.delegate_concurrent(alloc(100, 100, 1)).unwrap();
        let err = child.delegate_concurrent(alloc(10, 10, 1)).unwrap_err();
        assert!(matches!(err, BudgetExhausted::Depth));
    }

    #[test]
    fn agent_id_bootstrap_uuid_is_deterministic() {
        let a = AgentId::new("barnaby").bootstrap_uuid();
        let b = AgentId::new("barnaby").bootstrap_uuid();
        assert_eq!(a, b);
        assert_ne!(a, AgentId::new("diogenes").bootstrap_uuid());
    }

    #[test]
    fn agent_config_roundtrips_through_toml() {
        let cfg = AgentConfig {
            id: AgentId::new("barnaby"),
            model: ModelRef {
                provider: ModelProvider::Anthropic,
                model: "claude-opus-4-7".into(),
            },
            system_prompt: "You are Barnaby.".into(),
            memory_scope: MemoryScope::Private,
            allowed_capabilities: vec![CapabilityId("send_message".into())],
            default_budget: BudgetPolicy {
                input_tokens: 100_000,
                output_tokens: 20_000,
                calls: 50,
                max_depth: 3,
            },
            commit_policy: CommitPolicy::Periodic { interval: Duration::from_secs(300) },
        };

        let s = toml::to_string(&cfg).unwrap();
        let parsed: AgentConfig = toml::from_str(&s).unwrap();
        assert_eq!(parsed.id, cfg.id);
        assert_eq!(parsed.model.model, "claude-opus-4-7");
    }
}
