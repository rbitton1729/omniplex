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
use omniplex::config::DaemonConfig;
use omniplex::daemon::prepare;
use omniplex::harness::AgentHarness;
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

    // Validate everything (configs, models, name uniqueness) before
    // touching the filesystem or binding any socket.
    let plan = prepare(&daemon, &catalog, registry.as_ref())?;

    std::fs::create_dir_all(&plan.socket_dir)?;

    let mut set: JoinSet<()> = JoinSet::new();
    for agent in plan.agents {
        // Clear a stale socket from a prior (possibly crashed) run.
        if agent.socket_path.exists() {
            std::fs::remove_file(&agent.socket_path)?;
        }

        let harness = AgentHarness::bind((*agent.config).clone(), agent.socket_path)?;
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
