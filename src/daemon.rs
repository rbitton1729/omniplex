//! Daemon orchestration helpers.
//!
//! The `omniplexd` binary is a thin wrapper: load config, prepare a plan,
//! bind sockets, run accept loops. This module owns the "load + prepare"
//! half so it can be exercised by unit tests without binding sockets or
//! touching the runtime.
//!
//! [`prepare`] performs every check that should fail fast:
//!   - load each referenced agent TOML,
//!   - look up the agent's model in the catalog,
//!   - insert the agent into the registry (which rejects duplicates),
//!   - compute the well-known socket path.
//!
//! On success, callers walk the returned [`DaemonPlan`] to bind sockets
//! and spawn harnesses. On failure, no socket has been touched.

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;
use tokio::task::JoinSet;
use uuid::Uuid;

use crate::bus::socket_path_for;
use crate::catalog::{CatalogError, ModelCatalog};
use crate::config::{DaemonConfig, load_agent_config};
use crate::executor::Executor;
use crate::harness::{AgentHarness, HarnessError};
use crate::types::{AgentConfig, Registry, RegistryError};

#[derive(Debug, Error)]
pub enum DaemonPrepareError {
    #[error("failed to load agent config at {path}: {message}")]
    AgentConfig { path: PathBuf, message: String },

    #[error("agent {agent} references unknown model: {source}")]
    UnknownModel {
        agent: String,
        #[source]
        source: CatalogError,
    },

    #[error("duplicate agent name in daemon config: {0}")]
    DuplicateAgent(String),

    #[error("registry rejected agent {agent}: {source}")]
    Registry {
        agent: String,
        #[source]
        source: RegistryError,
    },
}

/// One agent's prepared state: its registry-canonical config, its UUID,
/// and the socket path it should bind.
#[derive(Debug, Clone)]
pub struct AgentPlan {
    pub uuid: Uuid,
    pub config: Arc<AgentConfig>,
    pub socket_path: PathBuf,
}

/// Result of [`prepare`]: validated socket directory, per-agent plans, and
/// the shared executor each bound harness will dispatch through.
#[derive(Debug, Clone)]
pub struct DaemonPlan {
    pub socket_dir: PathBuf,
    pub agents: Vec<AgentPlan>,
    pub executor: Arc<dyn Executor>,
}

impl DaemonPlan {
    /// Swap the executor without re-running `prepare`. Useful in tests that
    /// want to reuse a built plan with a different implementation.
    pub fn with_executor(mut self, executor: Arc<dyn Executor>) -> Self {
        self.executor = executor;
        self
    }
}

/// Load every agent referenced by `daemon`, validate models against
/// `catalog`, and insert each into `registry`. Returns a plan describing
/// what the daemon should bind. Does not touch the socket directory.
pub fn prepare(
    daemon: &DaemonConfig,
    catalog: &ModelCatalog,
    registry: &dyn Registry,
    executor: Arc<dyn Executor>,
) -> Result<DaemonPlan, DaemonPrepareError> {
    let mut agents = Vec::with_capacity(daemon.agent_paths.len());

    for path in &daemon.agent_paths {
        let cfg = load_agent_config(path).map_err(|e| DaemonPrepareError::AgentConfig {
            path: path.clone(),
            message: e.to_string(),
        })?;

        catalog
            .lookup(&cfg.model)
            .map_err(|source| DaemonPrepareError::UnknownModel {
                agent: cfg.id.name.clone(),
                source,
            })?;

        let name = cfg.id.name.clone();
        let socket_path = socket_path_for(&daemon.socket_dir, &name);
        let uuid = registry.insert(cfg).map_err(|source| match source {
            RegistryError::AlreadyExists(n) => DaemonPrepareError::DuplicateAgent(n),
            other => DaemonPrepareError::Registry {
                agent: name.clone(),
                source: other,
            },
        })?;
        let config = registry
            .get(&name)
            .map_err(|source| DaemonPrepareError::Registry {
                agent: name.clone(),
                source,
            })?;

        agents.push(AgentPlan {
            uuid,
            config,
            socket_path,
        });
    }

    Ok(DaemonPlan {
        socket_dir: daemon.socket_dir.clone(),
        agents,
        executor,
    })
}

#[derive(Debug, Error)]
pub enum DaemonBindError {
    #[error("failed to create socket dir {path}: {source}")]
    CreateSocketDir {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to remove stale socket {path}: {source}")]
    RemoveStaleSocket {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to bind {path}: {source}")]
    Bind {
        path: PathBuf,
        #[source]
        source: HarnessError,
    },
}

/// One agent bound to its Unix socket and ready to serve.
pub struct BoundAgent {
    pub uuid: Uuid,
    pub config: Arc<AgentConfig>,
    pub socket_path: PathBuf,
    harness: AgentHarness,
}

/// Result of [`bind_plan`]: socket directory created, every agent's socket
/// bound, ready to spawn the accept loops.
pub struct BoundDaemon {
    pub socket_dir: PathBuf,
    pub agents: Vec<BoundAgent>,
}

/// Read-only metadata about a running agent — what callers need to log or
/// route to after the harness has been spawned onto a task.
#[derive(Debug, Clone)]
pub struct AgentInfo {
    pub name: String,
    pub socket_path: PathBuf,
}

/// Materialize a [`DaemonPlan`] on disk: create the socket directory,
/// clear any stale socket file at each agent's well-known path, and bind
/// the per-agent [`AgentHarness`]. Returns the bound harnesses so callers
/// can either drive `spawn_all` or drop to tear everything down.
pub fn bind_plan(plan: DaemonPlan) -> Result<BoundDaemon, DaemonBindError> {
    let DaemonPlan {
        socket_dir,
        agents: plan_agents,
        executor,
    } = plan;

    std::fs::create_dir_all(&socket_dir).map_err(|source| DaemonBindError::CreateSocketDir {
        path: socket_dir.clone(),
        source,
    })?;

    let mut agents = Vec::with_capacity(plan_agents.len());
    for agent in plan_agents {
        // Clear a stale socket from a prior (possibly crashed) run, else
        // UnixListener::bind would fail with AddrInUse.
        if agent.socket_path.exists() {
            std::fs::remove_file(&agent.socket_path).map_err(|source| {
                DaemonBindError::RemoveStaleSocket {
                    path: agent.socket_path.clone(),
                    source,
                }
            })?;
        }

        let harness = AgentHarness::bind(
            (*agent.config).clone(),
            agent.socket_path.clone(),
            Some(executor.clone()),
        )
        .map_err(|source| DaemonBindError::Bind {
            path: agent.socket_path.clone(),
            source,
        })?;

        agents.push(BoundAgent {
            uuid: agent.uuid,
            config: agent.config,
            socket_path: agent.socket_path,
            harness,
        });
    }

    Ok(BoundDaemon { socket_dir, agents })
}

impl BoundDaemon {
    /// Consume each bound agent and spawn its accept loop onto a [`JoinSet`].
    /// Returns the join set alongside stable per-agent metadata so the caller
    /// can log, shut down, or match replies back to agents.
    pub fn spawn_all(self) -> (JoinSet<()>, Vec<AgentInfo>) {
        let mut set = JoinSet::new();
        let mut infos = Vec::with_capacity(self.agents.len());
        for agent in self.agents {
            infos.push(AgentInfo {
                name: agent.config.id.name.clone(),
                socket_path: agent.socket_path.clone(),
            });
            let name = agent.config.id.name.clone();
            set.spawn(async move {
                if let Err(e) = agent.harness.run().await {
                    eprintln!("harness[{name}] exited with error: {e}");
                }
            });
        }
        (set, infos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::{PeerMessage, PeerReply, send_one, socket_path_for};
    use crate::executor::StubExecutor;
    use crate::registry::InMemoryRegistry;
    use crate::types::{
        AgentConfig, AgentId, BudgetPolicy, CommitPolicy, DeferCondition, Deliberation,
        MemoryScope, ModelProvider, ModelRef,
    };
    use std::path::Path;
    use tempfile::TempDir;

    fn stub_executor() -> Arc<dyn Executor> {
        Arc::new(StubExecutor)
    }

    fn agent_config(name: &str, model: &str) -> AgentConfig {
        AgentConfig {
            id: AgentId::new(name),
            model: ModelRef {
                provider: ModelProvider::Anthropic,
                model: model.into(),
            },
            system_prompt: format!("You are {name}."),
            memory_scope: MemoryScope::Private,
            allowed_capabilities: Vec::new(),
            default_budget: BudgetPolicy {
                input_tokens: 1_000,
                output_tokens: 1_000,
                calls: 10,
                max_depth: 2,
            },
            commit_policy: CommitPolicy::Manual,
        }
    }

    fn write_agent(dir: &Path, file: &str, cfg: &AgentConfig) -> PathBuf {
        let path = dir.join(file);
        std::fs::write(&path, toml::to_string(cfg).unwrap()).unwrap();
        path
    }

    fn daemon_with_agents(socket_dir: PathBuf, agent_paths: Vec<PathBuf>) -> DaemonConfig {
        DaemonConfig {
            socket_dir,
            agent_paths,
        }
    }

    #[test]
    fn prepare_loads_validates_and_inserts_agents() {
        let tmp = TempDir::new().unwrap();
        let socket_dir = tmp.path().join("sockets");
        let a = write_agent(
            tmp.path(),
            "barnaby.toml",
            &agent_config("barnaby", "claude-opus-4-7"),
        );
        let b = write_agent(
            tmp.path(),
            "diogenes.toml",
            &agent_config("diogenes", "claude-sonnet-4-6"),
        );
        let daemon = daemon_with_agents(socket_dir.clone(), vec![a, b]);
        let catalog = ModelCatalog::with_builtin();
        let registry = InMemoryRegistry::default();

        let plan =
            prepare(&daemon, &catalog, &registry, stub_executor()).expect("prepare succeeds");

        assert_eq!(plan.socket_dir, socket_dir);
        assert_eq!(plan.agents.len(), 2);
        assert_eq!(plan.agents[0].config.id.name, "barnaby");
        assert_eq!(plan.agents[1].config.id.name, "diogenes");
        assert_eq!(
            plan.agents[0].socket_path,
            socket_path_for(&socket_dir, "barnaby")
        );
        assert_eq!(
            plan.agents[1].socket_path,
            socket_path_for(&socket_dir, "diogenes")
        );
        assert_eq!(
            plan.agents[0].uuid,
            AgentId::new("barnaby").bootstrap_uuid()
        );

        let mut listed = registry.list();
        listed.sort();
        assert_eq!(listed, vec!["barnaby".to_string(), "diogenes".to_string()]);
    }

    #[test]
    fn prepare_fails_on_unknown_model() {
        let tmp = TempDir::new().unwrap();
        let path = write_agent(
            tmp.path(),
            "ghost.toml",
            &agent_config("ghost", "claude-ghost-99"),
        );
        let daemon = daemon_with_agents(tmp.path().join("sockets"), vec![path]);
        let catalog = ModelCatalog::with_builtin();
        let registry = InMemoryRegistry::default();

        let err = prepare(&daemon, &catalog, &registry, stub_executor()).unwrap_err();
        assert!(
            matches!(err, DaemonPrepareError::UnknownModel { ref agent, .. } if agent == "ghost"),
            "unexpected error: {err:?}"
        );
        // Bad model must not pollute the registry.
        assert!(registry.list().is_empty());
    }

    #[test]
    fn prepare_fails_on_duplicate_agent_name() {
        let tmp = TempDir::new().unwrap();
        let a = write_agent(
            tmp.path(),
            "a.toml",
            &agent_config("barnaby", "claude-opus-4-7"),
        );
        let b = write_agent(
            tmp.path(),
            "b.toml",
            &agent_config("barnaby", "claude-sonnet-4-6"),
        );
        let daemon = daemon_with_agents(tmp.path().join("sockets"), vec![a, b]);
        let catalog = ModelCatalog::with_builtin();
        let registry = InMemoryRegistry::default();

        let err = prepare(&daemon, &catalog, &registry, stub_executor()).unwrap_err();
        assert!(
            matches!(err, DaemonPrepareError::DuplicateAgent(ref name) if name == "barnaby"),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn prepare_fails_when_agent_path_is_missing() {
        let tmp = TempDir::new().unwrap();
        let missing = tmp.path().join("nope.toml");
        let daemon = daemon_with_agents(tmp.path().join("sockets"), vec![missing.clone()]);
        let catalog = ModelCatalog::with_builtin();
        let registry = InMemoryRegistry::default();

        let err = prepare(&daemon, &catalog, &registry, stub_executor()).unwrap_err();
        assert!(
            matches!(err, DaemonPrepareError::AgentConfig { ref path, .. } if path == &missing),
            "unexpected error: {err:?}"
        );
    }

    fn plan_with_one_agent(
        tmp: &TempDir,
        socket_dir: PathBuf,
        name: &str,
    ) -> (DaemonPlan, InMemoryRegistry) {
        let agent_file = write_agent(
            tmp.path(),
            &format!("{name}.toml"),
            &agent_config(name, "claude-opus-4-7"),
        );
        let daemon = daemon_with_agents(socket_dir, vec![agent_file]);
        let catalog = ModelCatalog::with_builtin();
        let registry = InMemoryRegistry::default();
        let plan =
            prepare(&daemon, &catalog, &registry, stub_executor()).expect("prepare succeeds");
        (plan, registry)
    }

    #[tokio::test]
    async fn bind_plan_spawns_harness_that_answers_ping_and_status() {
        let tmp = TempDir::new().unwrap();
        let socket_dir = tmp.path().join("sockets");
        let (plan, _registry) = plan_with_one_agent(&tmp, socket_dir.clone(), "barnaby");

        let bound = bind_plan(plan).expect("bind_plan succeeds");

        assert!(socket_dir.is_dir(), "socket_dir should be created");
        assert_eq!(bound.socket_dir, socket_dir);
        assert_eq!(bound.agents.len(), 1);
        let expected_socket = socket_path_for(&socket_dir, "barnaby");
        assert_eq!(bound.agents[0].socket_path, expected_socket);
        assert_eq!(bound.agents[0].config.id.name, "barnaby");
        assert!(
            expected_socket.exists(),
            "bind_plan should leave the socket file in place"
        );

        let (mut set, infos) = bound.spawn_all();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].name, "barnaby");
        assert_eq!(infos[0].socket_path, expected_socket);

        let ack = send_one(&socket_dir, "barnaby", &PeerMessage::Ping)
            .await
            .expect("ping reaches harness");
        assert!(matches!(ack, PeerReply::Ack), "expected Ack, got {ack:?}");

        let status = send_one(&socket_dir, "barnaby", &PeerMessage::Status)
            .await
            .expect("status reaches harness");
        match status {
            PeerReply::Status { agent, .. } => assert_eq!(agent.name, "barnaby"),
            other => panic!("expected Status, got {other:?}"),
        }

        set.shutdown().await;
    }

    #[tokio::test]
    async fn bind_plan_replaces_stale_socket_file() {
        let tmp = TempDir::new().unwrap();
        let socket_dir = tmp.path().join("sockets");
        std::fs::create_dir_all(&socket_dir).unwrap();
        let stale = socket_path_for(&socket_dir, "barnaby");
        std::fs::write(&stale, b"stale leftover").unwrap();

        let (plan, _registry) = plan_with_one_agent(&tmp, socket_dir.clone(), "barnaby");

        let bound = bind_plan(plan).expect("bind_plan removes stale socket and binds");
        assert_eq!(bound.agents[0].socket_path, stale);

        // The listener is live now — a Ping round-trip proves the stale file
        // was not just overwritten but replaced by a real Unix socket.
        let (mut set, _) = bound.spawn_all();
        let reply = send_one(&socket_dir, "barnaby", &PeerMessage::Ping)
            .await
            .expect("harness answers after stale file was cleared");
        assert!(matches!(reply, PeerReply::Ack));
        set.shutdown().await;
    }

    #[tokio::test]
    async fn bind_plan_dispatches_message_through_stub_executor() {
        let tmp = TempDir::new().unwrap();
        let socket_dir = tmp.path().join("sockets");
        let (plan, _registry) = plan_with_one_agent(&tmp, socket_dir.clone(), "barnaby");
        let bound = bind_plan(plan).expect("bind_plan succeeds");

        let (mut set, _) = bound.spawn_all();

        let msg = PeerMessage::Message {
            from: crate::bus::PeerOrigin::External {
                label: "test".into(),
            },
            text: "hi".into(),
            sent_at: chrono::Utc::now(),
        };
        let reply = send_one(&socket_dir, "barnaby", &msg)
            .await
            .expect("message reaches harness");
        match reply {
            PeerReply::Deliberation(Deliberation::Defer { until, reasoning }) => {
                assert!(matches!(until, DeferCondition::Never));
                assert_eq!(reasoning, "executor not yet wired to model client");
            }
            other => panic!("expected stub Defer, got {other:?}"),
        }

        set.shutdown().await;
    }

    #[tokio::test]
    async fn bind_plan_creates_missing_socket_dir() {
        let tmp = TempDir::new().unwrap();
        let socket_dir = tmp.path().join("nested/sockets");
        assert!(!socket_dir.exists());
        let (plan, _registry) = plan_with_one_agent(&tmp, socket_dir.clone(), "barnaby");

        let bound = bind_plan(plan).expect("bind_plan creates the socket dir");

        assert!(socket_dir.is_dir());
        drop(bound);
    }
}
