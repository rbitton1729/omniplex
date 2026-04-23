//! Ollama HTTP chat client.
//!
//! Phase 1 scope: a single non-streaming POST to `/api/chat` that turns one
//! optional system prompt + one user message into one assistant string.
//! No streaming, no token accounting (Phase 2), no tool use, no embeddings.
//!
//! The client is intentionally narrow — keep the surface small until the
//! executor actually needs more from it. Multi-turn conversation and
//! per-call options (temperature, num_ctx, …) belong in a follow-up.

use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OllamaError {
    #[error("connection error: {0}")]
    Connection(String),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    #[error("unexpected status {status}: {body}")]
    Unexpected { status: u16, body: String },
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
}

#[derive(Debug, Clone, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Clone, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatResponseMessage {
    #[allow(dead_code)]
    role: String,
    content: String,
}

#[derive(Debug, Clone, Deserialize)]
struct ChatResponse {
    message: ChatResponseMessage,
}

#[derive(Debug, Clone)]
pub struct OllamaChatClient {
    base_url: String,
    model: String,
    http: reqwest::Client,
}

impl OllamaChatClient {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self::with_client(base_url, model, reqwest::Client::new())
    }

    pub fn with_client(
        base_url: impl Into<String>,
        model: impl Into<String>,
        http: reqwest::Client,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            http,
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// POST `/api/chat` with `stream: false` and return `message.content`.
    /// `system_message` is optional; when present it is prepended as the
    /// first message with `role: system`.
    pub async fn chat(
        &self,
        system_message: Option<&str>,
        user_message: &str,
    ) -> Result<String, OllamaError> {
        let mut messages = Vec::with_capacity(2);
        if let Some(sys) = system_message {
            messages.push(ChatMessage {
                role: "system",
                content: sys,
            });
        }
        messages.push(ChatMessage {
            role: "user",
            content: user_message,
        });

        let req = ChatRequest {
            model: &self.model,
            messages,
            stream: false,
        };

        let url = format!("{}/api/chat", self.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| OllamaError::Connection(e.to_string()))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(map_status_error(status, body));
        }

        let parsed: ChatResponse = resp
            .json()
            .await
            .map_err(|e| OllamaError::Connection(e.to_string()))?;
        Ok(parsed.message.content)
    }
}

fn map_status_error(status: StatusCode, body: String) -> OllamaError {
    match status {
        StatusCode::NOT_FOUND => OllamaError::ModelNotFound(body),
        StatusCode::UNAUTHORIZED => OllamaError::Unauthorized(body),
        s => OllamaError::Unexpected {
            status: s.as_u16(),
            body,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn test_deserialize_chat_response() {
        // Real-shape Ollama response with extra fields the client doesn't care about.
        let json = r#"{
            "model": "llama3:8b",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "hello world"
            },
            "done_reason": "stop",
            "done": true,
            "total_duration": 1234,
            "load_duration": 5,
            "prompt_eval_count": 10,
            "eval_count": 5
        }"#;
        let parsed: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.message.content, "hello world");
        assert_eq!(parsed.message.role, "assistant");
    }

    #[test]
    fn test_chat_request_builder() {
        let req = ChatRequest {
            model: "llama3:8b",
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: "be helpful",
                },
                ChatMessage {
                    role: "user",
                    content: "hi",
                },
            ],
            stream: false,
        };

        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["model"], "llama3:8b");
        assert_eq!(v["stream"], false);
        assert_eq!(v["messages"][0]["role"], "system");
        assert_eq!(v["messages"][0]["content"], "be helpful");
        assert_eq!(v["messages"][1]["role"], "user");
        assert_eq!(v["messages"][1]["content"], "hi");
    }

    #[tokio::test]
    async fn chat_returns_message_content_from_mock() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_json(serde_json::json!({
                "model": "llama3:8b",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "ping"}
                ],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "pong"},
                "done": true
            })))
            .mount(&server)
            .await;

        let client = OllamaChatClient::new(server.uri(), "llama3:8b");
        let out = client.chat(Some("sys"), "ping").await.unwrap();
        assert_eq!(out, "pong");
    }

    #[tokio::test]
    async fn chat_omits_system_message_when_none() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .and(body_json(serde_json::json!({
                "model": "llama3:8b",
                "messages": [
                    {"role": "user", "content": "hi"}
                ],
                "stream": false
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3:8b",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": "yo"},
                "done": true
            })))
            .mount(&server)
            .await;

        let client = OllamaChatClient::new(server.uri(), "llama3:8b");
        assert_eq!(client.chat(None, "hi").await.unwrap(), "yo");
    }

    #[tokio::test]
    async fn chat_404_maps_to_model_not_found() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(404).set_body_string("model not pulled"))
            .mount(&server)
            .await;

        let client = OllamaChatClient::new(server.uri(), "missing");
        let err = client.chat(None, "hi").await.unwrap_err();
        match err {
            OllamaError::ModelNotFound(body) => assert!(body.contains("model not pulled")),
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_401_maps_to_unauthorized() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(401).set_body_string("nope"))
            .mount(&server)
            .await;

        let client = OllamaChatClient::new(server.uri(), "any");
        let err = client.chat(None, "hi").await.unwrap_err();
        assert!(matches!(err, OllamaError::Unauthorized(_)));
    }

    #[tokio::test]
    async fn chat_other_status_maps_to_unexpected() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(503).set_body_string("overloaded"))
            .mount(&server)
            .await;

        let client = OllamaChatClient::new(server.uri(), "any");
        let err = client.chat(None, "hi").await.unwrap_err();
        match err {
            OllamaError::Unexpected { status, body } => {
                assert_eq!(status, 503);
                assert!(body.contains("overloaded"));
            }
            other => panic!("expected Unexpected, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_connection_failure_maps_to_connection_error() {
        // Bind a port and immediately drop it so connect fails.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let client = OllamaChatClient::new(format!("http://{addr}"), "any");
        let err = client.chat(None, "hi").await.unwrap_err();
        assert!(matches!(err, OllamaError::Connection(_)));
    }
}
