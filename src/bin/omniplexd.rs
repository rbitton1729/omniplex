//! Omniplex parent daemon.
//!
//! Reads a daemon TOML describing a socket directory and a list of agent
//! config paths, spawns one [`AgentHarness`] per agent, and waits for
//! Ctrl-C. Each harness binds `<socket_dir>/<name>.sock` and serves
//! `PeerMessage` requests one at a time.
//!
//! Phase 0 scope: config → spawn → accept loop. No hot reload, no
//! supervision (failed harnesses exit their task; the daemon does not
//! restart them), no watch scheduling. Models are validated against the
//! built-in catalog on startup so a typo in a TOML fails fast.
//!
//! Daemon config format:
//! ```toml
//! socket_dir = "/run/omniplex"
//! agents = ["configs/barnaby.toml", "configs/diogenes.toml"]
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use omniplex::bus::socket_path_for;
use omniplex::catalog::ModelCatalog;
use omniplex::harness::AgentHarness;
use omniplex::types::AgentConfig;
use serde::Deserialize;
use tokio::task::JoinSet;

#[derive(Debug, Deserialize)]
struct DaemonConfig {
    socket_dir: PathBuf,
    #[serde(default)]
    agents: Vec<PathBuf>,
}

fn usage() -> ! {
    eprintln!("usage: omniplexd <daemon-config.toml>");
    std::process::exit(2);
}

fn load_daemon_config(path: &Path) -> Result<DaemonConfig, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&raw)?)
}

fn load_agent_config(path: &Path) -> Result<AgentConfig, Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&raw)?)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let config_path = match args.get(1) {
        Some(p) => PathBuf::from(p),
        None => usage(),
    };

    let daemon = load_daemon_config(&config_path)?;
    std::fs::create_dir_all(&daemon.socket_dir)?;

    let catalog = Arc::new(ModelCatalog::with_builtin());

    let mut set: JoinSet<()> = JoinSet::new();
    for agent_path in &daemon.agents {
        let cfg = load_agent_config(agent_path)?;

        // Fail fast on unknown models rather than after spawn.
        catalog.lookup(&cfg.model)?;

        let socket_path = socket_path_for(&daemon.socket_dir, &cfg.id.name);
        // Clear a stale socket from a prior (possibly crashed) run.
        if socket_path.exists() {
            std::fs::remove_file(&socket_path)?;
        }

        let harness = AgentHarness::bind(cfg, socket_path)?;
        eprintln!(
            "omniplexd: spawned {} at {}",
            harness.agent_name(),
            harness.socket_path().display()
        );
        set.spawn(async move {
            if let Err(e) = harness.run().await {
                eprintln!("harness exited with error: {e}");
            }
        });
    }

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            eprintln!("omniplexd: Ctrl-C received, shutting down");
        }
        _ = wait_all(&mut set) => {
            eprintln!("omniplexd: all harness tasks exited");
        }
    }

    set.shutdown().await;
    Ok(())
}

async fn wait_all(set: &mut JoinSet<()>) {
    while set.join_next().await.is_some() {}
}
