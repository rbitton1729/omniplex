# Omniplex — Development Roadmap

## Phase 0: Foundation (IN PROGRESS)

- [x] Core types (`Budget`, `Reservation`, `AgentId`, `AgentConfig`, `Deliberation`, `Capability`)
- [x] ModelCatalog with builtin pricing
- [x] Lantern-backed MemoryService actor
- [x] Inter-agent Unix socket bus (`bus.rs`)
  - `PeerMessage` / `PeerReply` wire protocol
  - One-shot request/response over `UnixStream`
  - Agent discovery via well-known socket directory (`<socket_dir>/<name>.sock`)
  - Phase 2: SO_PEERCRED auth, streaming replies, broadcast
- [ ] Omniplexd parent process (spawn agents, supervise, signal handling)
- [ ] Agent harness (accept loop + recv_one + route to deliberation)
- [ ] Config loader + hot reload
- [ ] Empty `main.rs` → `omniplexd` binary entrypoint

## Phase 1: Simple Agents

- [ ] Deliberation executor (Act / Defer / Abandon dispatch)
- [ ] Model client bridge (Anthropic, Ollama, OpenAI-compatible)
- [ ] Watch scheduler (`Pattern` + `Once` triggers)
  - CI pipeline watches via GitLab webhook → socket event injection
- [ ] Barnaby end-to-end: quiet detection → deliberation → `send_telegram_message`
- [ ] Budget enforcement on real model calls
- [ ] Memory scope enforcement (Private / Shared / Global)

## Phase 2: Multi-Agent + Persistence

- [ ] Deferred task queue (time-based wakes)
- [ ] Agent-to-agent delegation via bus (`Delegate` message)
- [ ] Semantic search memory (Ollama `nomic-embed-text`)
- [ ] `NeedMore` deliberation variant with deliberative watches
- [ ] Streaming deliberation replies
- [ ] Commit policy implementations (Manual, OnCheckpoint, Periodic, OnShutdown)

## Phase 3: Production

- [ ] MCP client bridge for capabilities
- [ ] Process crash recovery (replay committed intent + exactly-once side effects)
- [ ] External event sources (Telegram bot, webhooks, cron)
- [ ] `omniplexctl` CLI: status, trace, audit, spawn, kill
- [ ] Multi-host (TCP mesh for agents across machines)

## Blockers

- `docs/lantern/` path dependency missing locally.  Options:
  1. Stub out minimal Lantern interface for harness development
  2. Add lantern as a proper submodule / git dep / workspace member
  3. Use feature gate to make Lantern optional in Phase 1
