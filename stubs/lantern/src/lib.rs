//! Stub for the real `lantern` crate.
//!
//! Only the types and function signatures referenced by `omniplex::memory`
//! are provided. Operations return empty results or a stub error so the
//! project compiles; runtime behaviour is deliberately useless until the
//! real crate is vendored back into the tree.

pub mod store {
    use std::path::Path;

    #[derive(Debug)]
    pub struct Store;

    #[derive(Debug)]
    pub struct StoreError(pub String);

    impl std::fmt::Display for StoreError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl std::error::Error for StoreError {}

    impl Store {
        pub fn initialize(_root: &Path) -> Result<Self, StoreError> {
            Err(StoreError(
                "lantern stub: Store::initialize is not implemented".to_string(),
            ))
        }
    }
}

pub mod ingest {
    use super::store::{Store, StoreError};

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
        _store: &mut Store,
        _uri: &str,
        _mime: Option<&str>,
        _bytes: &[u8],
    ) -> Result<IngestReport, StoreError> {
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
        _store: &Store,
        _query: &str,
        _opts: SearchOptions,
    ) -> Result<Vec<SearchHit>, StoreError> {
        Ok(Vec::new())
    }
}
