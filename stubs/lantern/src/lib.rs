//! Stub for the real `lantern` crate.
//!
//! Only the types and function signatures referenced by `omniplex::memory`
//! are provided. This stub implements a tiny in-process JSONL store so the
//! Omniplex memory actor can be exercised in tests until the real crate is
//! vendored back into the tree.

pub mod store {
    use std::path::Path;

    #[derive(Debug)]
    pub struct Store {
        pub(crate) records: Vec<crate::Record>,
    }

    #[derive(Debug)]
    pub struct StoreError(pub String);

    impl std::fmt::Display for StoreError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl std::error::Error for StoreError {}

    impl Store {
        pub fn initialize(root: &Path) -> Result<Self, StoreError> {
            std::fs::create_dir_all(root).map_err(|e| StoreError(e.to_string()))?;
            Ok(Self { records: Vec::new() })
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Record {
    uri: String,
    ordinal: i64,
    text: String,
    role: Option<String>,
    session_id: Option<String>,
    turn_id: Option<String>,
}

pub mod ingest {
    use serde_json::Value;

    use super::{store::{Store, StoreError}, Record};

    #[derive(Debug, Default)]
    pub struct IngestReport {
        pub skipped: Vec<SkippedSource>,
    }

    #[derive(Debug)]
    pub struct SkippedSource {
        pub uri: String,
        pub skipped_reason: Option<String>,
        pub reason: String,
    }

    pub fn ingest_stdin(
        store: &mut Store,
        uri: &str,
        _mime: Option<&str>,
        bytes: &[u8],
    ) -> Result<IngestReport, StoreError> {
        let input = std::str::from_utf8(bytes).map_err(|e| StoreError(e.to_string()))?;
        let mut records = Vec::new();

        for (idx, line) in input.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line).map_err(|e| StoreError(e.to_string()))?;
            let content = value
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or(line)
                .to_string();
            let role = value.get("role").and_then(Value::as_str).map(str::to_string);
            let session_id = value.get("session_id").and_then(Value::as_str).map(str::to_string);
            let turn_id = value.get("turn_id").and_then(Value::as_str).map(str::to_string);

            records.push(Record {
                uri: uri.to_string(),
                ordinal: idx as i64,
                text: content,
                role,
                session_id,
                turn_id,
            });
        }

        store.records.retain(|record| record.uri != uri);
        store.records.extend(records);
        Ok(IngestReport::default())
    }
}

pub mod search {
    use super::store::{Store, StoreError};

    #[derive(Debug, Default, Clone)]
    pub struct SearchOptions {
        pub limit: usize,
        pub kind: Option<String>,
        pub path_contains: Option<String>,
    }

    #[derive(Debug, Clone)]
    pub struct SearchHit {
        pub uri: String,
        pub ordinal: i64,
        pub text: String,
        pub snippet: String,
        pub score: f64,
        pub role: Option<String>,
        pub session_id: Option<String>,
        pub turn_id: Option<String>,
    }

    pub fn search(
        store: &Store,
        query: &str,
        opts: SearchOptions,
    ) -> Result<Vec<SearchHit>, StoreError> {
        let needle = query.to_lowercase();
        let mut hits: Vec<_> = store
            .records
            .iter()
            .filter(|record| {
                if let Some(path_contains) = &opts.path_contains {
                    if !record.uri.contains(path_contains) {
                        return false;
                    }
                }
                needle.is_empty() || record.text.to_lowercase().contains(&needle)
            })
            .take(opts.limit)
            .map(|record| SearchHit {
                uri: record.uri.clone(),
                ordinal: record.ordinal,
                text: record.text.clone(),
                snippet: record.text.clone(),
                score: 1.0,
                role: record.role.clone(),
                session_id: record.session_id.clone(),
                turn_id: record.turn_id.clone(),
            })
            .collect();

        hits.sort_by(|left, right| left.ordinal.cmp(&right.ordinal));
        Ok(hits)
    }
}
