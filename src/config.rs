use std::path::{Component, Path, PathBuf};

use serde::Deserialize;

use crate::types::AgentConfig;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonConfig {
    pub socket_dir: PathBuf,
    pub data_dir: PathBuf,
    pub agent_paths: Vec<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct RawDaemonConfig {
    socket_dir: PathBuf,
    data_dir: PathBuf,
    #[serde(default)]
    agents: Vec<PathBuf>,
}

impl DaemonConfig {
    pub fn load_from_file(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let raw = std::fs::read_to_string(path)?;
        let raw: RawDaemonConfig = toml::from_str(&raw)?;
        Ok(Self::from_raw(
            path.parent().unwrap_or_else(|| Path::new(".")),
            raw,
        ))
    }

    fn from_raw(base_dir: &Path, raw: RawDaemonConfig) -> Self {
        Self {
            socket_dir: resolve_path(base_dir, raw.socket_dir),
            data_dir: resolve_path(base_dir, raw.data_dir),
            agent_paths: raw
                .agents
                .into_iter()
                .map(|path| resolve_path(base_dir, path))
                .collect(),
        }
    }
}

pub fn load_agent_config(path: &Path) -> Result<AgentConfig, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&raw)?)
}

fn resolve_path(base_dir: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        normalize_path(path)
    } else {
        normalize_path(base_dir.join(path))
    }
}

fn normalize_path(path: PathBuf) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::RootDir | Component::Prefix(_) | Component::Normal(_) => {
                out.push(component.as_os_str());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn daemon_config_resolves_relative_agent_paths_from_config_dir() {
        let tmp = TempDir::new().unwrap();
        let conf_dir = tmp.path().join("conf");
        std::fs::create_dir_all(conf_dir.join("agents")).unwrap();
        let daemon_path = conf_dir.join("daemon.toml");
        std::fs::write(
            &daemon_path,
            r#"
socket_dir = "../run/sockets"
data_dir = "../run/agents"
agents = ["agents/barnaby.toml"]
"#,
        )
        .unwrap();

        let cfg = DaemonConfig::load_from_file(&daemon_path).unwrap();

        assert_eq!(cfg.socket_dir, tmp.path().join("run/sockets"));
        assert_eq!(cfg.data_dir, tmp.path().join("run/agents"));
        assert_eq!(cfg.agent_paths, vec![conf_dir.join("agents/barnaby.toml")]);
    }

    #[test]
    fn daemon_config_preserves_absolute_paths() {
        let tmp = TempDir::new().unwrap();
        let daemon_path = tmp.path().join("daemon.toml");
        let socket_dir = tmp.path().join("sockets");
        let data_dir = tmp.path().join("data");
        let agent_path = tmp.path().join("agents/barnaby.toml");
        std::fs::write(
            &daemon_path,
            format!(
                "socket_dir = {:?}\ndata_dir = {:?}\nagents = [{:?}]\n",
                socket_dir.to_string_lossy(),
                data_dir.to_string_lossy(),
                agent_path.to_string_lossy()
            ),
        )
        .unwrap();

        let cfg = DaemonConfig::load_from_file(&daemon_path).unwrap();

        assert_eq!(cfg.socket_dir, socket_dir);
        assert_eq!(cfg.data_dir, data_dir);
        assert_eq!(cfg.agent_paths, vec![agent_path]);
    }
}
