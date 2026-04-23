use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use uuid::Uuid;

use crate::types::{AgentConfig, AgentId, Registry, RegistryError};

#[derive(Debug, Default)]
pub struct InMemoryRegistry {
    inner: Mutex<RegistryState>,
}

#[derive(Debug, Default)]
struct RegistryState {
    agents: HashMap<String, RegisteredAgent>,
    aliases: HashMap<String, String>,
}

#[derive(Debug, Clone)]
struct RegisteredAgent {
    uuid: Uuid,
    config: Arc<AgentConfig>,
}

impl InMemoryRegistry {
    fn resolve<'a>(state: &'a RegistryState, name: &'a str) -> Option<&'a str> {
        if state.agents.contains_key(name) {
            Some(name)
        } else {
            state.aliases.get(name).map(String::as_str)
        }
    }
}

impl Registry for InMemoryRegistry {
    fn get(&self, name: &str) -> Result<Arc<AgentConfig>, RegistryError> {
        let state = self.inner.lock().expect("registry mutex poisoned");
        let canonical =
            Self::resolve(&state, name).ok_or_else(|| RegistryError::NotFound(name.to_string()))?;
        Ok(state
            .agents
            .get(canonical)
            .expect("resolved agent must exist")
            .config
            .clone())
    }

    fn uuid_of(&self, name: &str) -> Result<Uuid, RegistryError> {
        let state = self.inner.lock().expect("registry mutex poisoned");
        let canonical =
            Self::resolve(&state, name).ok_or_else(|| RegistryError::NotFound(name.to_string()))?;
        Ok(state
            .agents
            .get(canonical)
            .expect("resolved agent must exist")
            .uuid)
    }

    fn insert(&self, config: AgentConfig) -> Result<Uuid, RegistryError> {
        let mut state = self.inner.lock().expect("registry mutex poisoned");
        let name = config.id.name.clone();
        if state.agents.contains_key(&name) {
            return Err(RegistryError::AlreadyExists(name));
        }
        if state.aliases.contains_key(&name) {
            return Err(RegistryError::AliasCollision(name));
        }

        let uuid = config.id.bootstrap_uuid();
        state.agents.insert(
            name,
            RegisteredAgent {
                uuid,
                config: Arc::new(config),
            },
        );
        Ok(uuid)
    }

    fn rename(&self, from: &str, to: &str) -> Result<(), RegistryError> {
        let mut state = self.inner.lock().expect("registry mutex poisoned");
        if state.agents.contains_key(to) || state.aliases.contains_key(to) {
            return Err(RegistryError::AliasCollision(to.to_string()));
        }

        let canonical = Self::resolve(&state, from)
            .ok_or_else(|| RegistryError::NotFound(from.to_string()))?
            .to_string();
        let registered = state
            .agents
            .remove(&canonical)
            .expect("resolved agent must exist");

        let mut config = (*registered.config).clone();
        config.id = AgentId::new(to);
        state.agents.insert(
            to.to_string(),
            RegisteredAgent {
                uuid: registered.uuid,
                config: Arc::new(config),
            },
        );
        state.aliases.insert(canonical.clone(), to.to_string());
        if from != canonical {
            state.aliases.insert(from.to_string(), to.to_string());
        }
        Ok(())
    }

    fn list(&self) -> Vec<String> {
        let state = self.inner.lock().expect("registry mutex poisoned");
        let mut names: Vec<_> = state.agents.keys().cloned().collect();
        names.sort();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AgentConfig, AgentId, BudgetPolicy, CommitPolicy, MemoryScope, ModelProvider, ModelRef,
        Registry,
    };

    fn config(name: &str) -> AgentConfig {
        AgentConfig {
            id: AgentId::new(name),
            model: ModelRef {
                provider: ModelProvider::Anthropic,
                model: "claude-opus-4-7".into(),
            },
            system_prompt: format!("You are {name}"),
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

    #[test]
    fn insert_assigns_deterministic_uuid_and_get_returns_config() {
        let registry = InMemoryRegistry::default();
        let cfg = config("barnaby");

        let uuid = registry.insert(cfg.clone()).unwrap();

        assert_eq!(uuid, AgentId::new("barnaby").bootstrap_uuid());
        assert_eq!(registry.uuid_of("barnaby").unwrap(), uuid);
        assert_eq!(registry.get("barnaby").unwrap().id.name, "barnaby");
        assert_eq!(registry.list(), vec!["barnaby".to_string()]);
    }

    #[test]
    fn rename_preserves_uuid_and_leaves_old_name_as_alias() {
        let registry = InMemoryRegistry::default();
        let uuid = registry.insert(config("barnaby")).unwrap();

        registry.rename("barnaby", "diogenes").unwrap();

        assert_eq!(registry.uuid_of("barnaby").unwrap(), uuid);
        assert_eq!(registry.uuid_of("diogenes").unwrap(), uuid);
        assert_eq!(registry.get("barnaby").unwrap().id.name, "diogenes");
        assert_eq!(registry.list(), vec!["diogenes".to_string()]);
    }

    #[test]
    fn insert_and_rename_reject_collisions() {
        let registry = InMemoryRegistry::default();
        registry.insert(config("barnaby")).unwrap();
        registry.insert(config("diogenes")).unwrap();

        assert!(registry.insert(config("barnaby")).is_err());
        assert!(registry.rename("barnaby", "diogenes").is_err());
    }
}
