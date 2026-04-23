//! `omniplexctl`: a small CLI for poking the daemon over the bus.
//!
//! Usage:
//!   omniplexctl send   <agent-name> <message> [--config <daemon.toml>]
//!   omniplexctl status <agent-name>            [--config <daemon.toml>]
//!
//! Reads the daemon TOML to discover `socket_dir`, then talks to the named
//! agent's socket via [`omniplex::bus::send_one`]. The `PeerReply` is printed
//! as pretty JSON so it composes with `jq`.

use std::path::PathBuf;
use std::process::ExitCode;

use omniplex::bus::{PeerMessage, PeerOrigin, send_one};
use omniplex::config::DaemonConfig;

const DEFAULT_CONFIG: &str = "daemon.toml";

fn usage() -> ! {
    eprintln!(
        "usage:\n  \
         omniplexctl send   <agent-name> <message> [--config <daemon.toml>]\n  \
         omniplexctl status <agent-name>            [--config <daemon.toml>]"
    );
    std::process::exit(2);
}

#[derive(Debug)]
enum Command {
    Send {
        agent: String,
        text: String,
        config: PathBuf,
    },
    Status {
        agent: String,
        config: PathBuf,
    },
}

fn parse_args(mut args: Vec<String>) -> Command {
    if args.is_empty() {
        usage();
    }
    let cmd = args.remove(0);

    let mut config = PathBuf::from(DEFAULT_CONFIG);
    let mut positional = Vec::new();
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        if arg == "--config" {
            match iter.next() {
                Some(v) => config = PathBuf::from(v),
                None => usage(),
            }
        } else if let Some(v) = arg.strip_prefix("--config=") {
            config = PathBuf::from(v);
        } else {
            positional.push(arg);
        }
    }

    match cmd.as_str() {
        "send" => {
            if positional.len() < 2 {
                usage();
            }
            let agent = positional.remove(0);
            // Treat the remaining positionals as a single message body, joined
            // by spaces - saves the caller from quoting in the common case.
            let text = positional.join(" ");
            Command::Send {
                agent,
                text,
                config,
            }
        }
        "status" => {
            if positional.len() != 1 {
                usage();
            }
            let agent = positional.remove(0);
            Command::Status { agent, config }
        }
        _ => usage(),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = parse_args(args);

    let result = match command {
        Command::Send {
            agent,
            text,
            config,
        } => run_send(&config, &agent, &text).await,
        Command::Status { agent, config } => run_status(&config, &agent).await,
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("omniplexctl: {e}");
            ExitCode::from(1)
        }
    }
}

async fn run_send(
    config_path: &PathBuf,
    agent: &str,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let daemon = DaemonConfig::load_from_file(config_path)?;
    let msg = PeerMessage::Message {
        from: PeerOrigin::External {
            label: "omniplexctl".into(),
        },
        text: text.to_string(),
        sent_at: chrono::Utc::now(),
    };
    let reply = send_one(&daemon.socket_dir, agent, &msg).await?;
    println!("{}", serde_json::to_string_pretty(&reply)?);
    Ok(())
}

async fn run_status(
    config_path: &PathBuf,
    agent: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let daemon = DaemonConfig::load_from_file(config_path)?;
    let reply = send_one(&daemon.socket_dir, agent, &PeerMessage::Status).await?;
    println!("{}", serde_json::to_string_pretty(&reply)?);
    Ok(())
}
