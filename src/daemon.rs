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

use std::path::PathBuf;
use std::sync::Arc;

use thiserror::Error;
use uuid::Uuid;

use crate::bus::socket_path_for;
use crate::catalog::{CatalogError, ModelCatalog};
use crate::config::{DaemonConfig, load_agent_config};
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

/// Result of [`prepare`]: validated socket directory plus per-agent plans.
#[derive(Debug, Clone)]
pub struct DaemonPlan {
    pub socket_dir: PathBuf,
    pub agents: Vec<AgentPlan>,
}

/// Load every agent referenced by `daemon`, validate models against
/// `catalog`, and insert each into `registry`. Returns a plan describing
/// what the daemon should bind. Does not touch the socket directory.
pub fn prepare(
    daemon: &DaemonConfig,
    catalog: &ModelCatalog,
    registry: &dyn Registry,
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::InMemoryRegistry;
    use crate::types::{
        AgentConfig, AgentId, BudgetPolicy, CommitPolicy, MemoryScope, ModelProvider, ModelRef,
    };
    use std::path::Path;
    use tempfile::TempDir;

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

        let plan = prepare(&daemon, &catalog, &registry).expect("prepare succeeds");

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

        let err = prepare(&daemon, &catalog, &registry).unwrap_err();
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

        let err = prepare(&daemon, &catalog, &registry).unwrap_err();
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

        let err = prepare(&daemon, &catalog, &registry).unwrap_err();
        assert!(
            matches!(err, DaemonPrepareError::AgentConfig { ref path, .. } if path == &missing),
            "unexpected error: {err:?}"
        );
    }
}
