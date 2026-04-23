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

use std::path::PathBuf;
use std::sync::Arc;

use omniplex::catalog::ModelCatalog;
use omniplex::config::{DaemonConfig, load_agent_config};
use omniplex::daemon::{bind_plan, prepare};
use omniplex::executor::{Executor, StubExecutor, executor_for};
use omniplex::registry::InMemoryRegistry;
use tokio::task::JoinSet;

fn usage() -> ! {
    eprintln!("usage: omniplexd <daemon-config.toml>");
    std::process::exit(2);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let config_path = match args.get(1) {
        Some(p) => PathBuf::from(p),
        None => usage(),
    };

    let daemon = DaemonConfig::load_from_file(&config_path)?;
    let catalog = ModelCatalog::with_builtin();
    let registry = Arc::new(InMemoryRegistry::default());

    // Phase 1 daemon shares one executor across every harness, so pick it
    // from the first agent's model. Heterogeneous-provider daemons land
    // when per-agent executor wiring does.
    let executor: Arc<dyn Executor> = match daemon.agent_paths.first() {
        Some(path) => executor_for(&load_agent_config(path)?.model),
        None => Arc::new(StubExecutor),
    };

    // Validate everything (configs, models, name uniqueness) before
    // touching the filesystem or binding any socket.
    let plan = prepare(&daemon, &catalog, registry.as_ref(), executor)?;
    let bound = bind_plan(plan)?;

    let (mut set, infos) = bound.spawn_all();
    for info in &infos {
        eprintln!(
            "omniplexd: spawned {} at {}",
            info.name,
            info.socket_path.display()
        );
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
