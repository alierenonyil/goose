use super::api_client::{ApiClient, AuthMethod};
use super::base::{ConfigKey, MessageStream, Provider, ProviderDef, ProviderMetadata};
use super::openai_compatible::{handle_response, OpenAiCompatibleProvider};
use crate::conversation::message::Message;
use anyhow::Result;
use async_trait::async_trait;
use futures::future::BoxFuture;
use goose_providers::errors::ProviderError;
use goose_providers::model::ModelConfig;
use rmcp::model::Tool;
use serde_json::Value;

const ONYSOFT_PROVIDER_NAME: &str = "onysoft";
pub const ONYSOFT_API_HOST: &str = "https://api.onysoft.com/v1";
pub const ONYSOFT_DEFAULT_MODEL: &str = "anthropic/claude-sonnet-5.5";
pub const ONYSOFT_KNOWN_MODELS: &[&str] = &[
    "anthropic/claude-haiku-5.5",
    "anthropic/claude-opus-5",
    "anthropic/claude-sonnet-5.5",
    "deepseek/deepseek-v4-pro",
    "google/gemini-3.6-flash",
    "openai/gpt-5-pro",
    "openai/gpt-5.6-terra",
    "x-ai/grok-4.7",
];
pub const ONYSOFT_DOC_URL: &str = "https://api.onysoft.com/docs";
pub const ONYSOFT_BILLING_URL: &str = "https://api.onysoft.com/dashboard/billing";

/// `GET /v1/models` returns Onysoft's own envelope by default, which nests the catalog under
/// `data.models`. `format=openai` selects the OpenAI-shaped top-level `data` array.
const ONYSOFT_MODELS_PATH: &str = "models?format=openai";

/// Catalog entry types that `POST /v1/chat/completions` serves. The same catalog also lists
/// video, image, music, tts, stt, embedding and rerank models, which that endpoint rejects with
/// HTTP 400 `wrong_endpoint`, so they are dropped before they can reach the model picker.
const ONYSOFT_CHAT_MODEL_TYPES: [&str; 2] = ["chat", "multimodal"];

pub struct OnysoftProvider {
    inner: OpenAiCompatibleProvider,
}

impl OnysoftProvider {
    /// Onysoft bills every model from one prepaid balance, so a 402 is always recoverable by
    /// topping that balance up. Point the user straight at the page that does it.
    fn with_top_up_url(err: ProviderError) -> ProviderError {
        match err {
            ProviderError::CreditsExhausted { details, .. } => ProviderError::CreditsExhausted {
                details,
                top_up_url: Some(ONYSOFT_BILLING_URL.to_string()),
            },
            other => other,
        }
    }
}

/// Collect the ids of the catalog entries goose can actually drive, sorted.
fn chat_model_ids(catalog: &Value) -> Result<Vec<String>, ProviderError> {
    if let Some(message) = catalog
        .get("error")
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str)
    {
        return Err(ProviderError::Authentication(message.to_string()));
    }

    let entries = catalog
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ProviderError::RequestFailed("Missing 'data' array in models response".to_string())
        })?;

    let mut models: Vec<String> = entries
        .iter()
        .filter(|entry| {
            entry
                .get("model_type")
                .and_then(Value::as_str)
                .is_some_and(|model_type| ONYSOFT_CHAT_MODEL_TYPES.contains(&model_type))
        })
        .filter_map(|entry| entry.get("id").and_then(Value::as_str).map(str::to_string))
        .collect();
    models.sort();
    Ok(models)
}

impl goose_providers::base::ProviderDescriptor for OnysoftProvider {
    fn metadata() -> ProviderMetadata {
        ProviderMetadata::new(
            ONYSOFT_PROVIDER_NAME,
            "Onysoft",
            "Gateway with one OpenAI-compatible API, a single balance, and models from many providers",
            ONYSOFT_DEFAULT_MODEL,
            ONYSOFT_KNOWN_MODELS.to_vec(),
            ONYSOFT_DOC_URL,
            vec![
                ConfigKey::new("ONYSOFT_API_KEY", true, true, None, true),
                ConfigKey::new("ONYSOFT_HOST", false, false, Some(ONYSOFT_API_HOST), false),
            ],
        )
        .with_setup(
            crate::providers::catalog::ProviderSetupMetadata::api_key(
                crate::providers::catalog::ProviderSetupGroup::Default,
            )
            .with_docs_url(ONYSOFT_DOC_URL),
        )
    }
}

impl ProviderDef for OnysoftProvider {
    type Provider = Self;

    fn from_env(
        _extensions: Vec<crate::config::ExtensionConfig>,
        tls_config: Option<crate::providers::api_client::TlsConfig>,
    ) -> BoxFuture<'static, Result<Self>> {
        Box::pin(async move {
            let config = crate::config::Config::global();
            let api_key: String = config.get_secret("ONYSOFT_API_KEY")?;
            let host: String = config
                .get_param("ONYSOFT_HOST")
                .unwrap_or_else(|_| ONYSOFT_API_HOST.to_string());

            let api_client =
                ApiClient::new_with_tls(host, AuthMethod::BearerToken(api_key), tls_config)?
                    .with_request_builder(crate::session_context::session_id_request_builder());

            Ok(Self {
                inner: OpenAiCompatibleProvider::new(
                    ONYSOFT_PROVIDER_NAME.to_string(),
                    api_client,
                    String::new(),
                ),
            })
        })
    }
}

#[async_trait]
impl Provider for OnysoftProvider {
    fn get_name(&self) -> &str {
        self.inner.get_name()
    }

    async fn refresh_credentials(&self) -> Result<(), ProviderError> {
        self.inner.refresh_credentials().await
    }

    async fn stream(
        &self,
        model_config: &ModelConfig,
        system: &str,
        messages: &[Message],
        tools: &[Tool],
    ) -> Result<MessageStream, ProviderError> {
        self.inner
            .stream(model_config, system, messages, tools)
            .await
            .map_err(Self::with_top_up_url)
    }

    async fn fetch_supported_models(&self) -> Result<Vec<String>, ProviderError> {
        let response = self
            .inner
            .api_client()
            .response_get(ONYSOFT_MODELS_PATH)
            .await
            .map_err(|e| ProviderError::RequestFailed(e.to_string()))?;
        chat_model_ids(&handle_response(response).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn catalog() -> Value {
        json!({
            "success": true,
            "object": "list",
            "data": [
                {"id": "x-ai/grok-4.7", "object": "model", "owned_by": "x-ai", "model_type": "chat"},
                {"id": "anthropic/claude-sonnet-5.5", "object": "model", "owned_by": "anthropic", "model_type": "chat"},
                {"id": "google/gemini-3.6-flash", "object": "model", "owned_by": "google", "model_type": "multimodal"},
                {"id": "vidu-q4/image-to-video-1080p", "object": "model", "owned_by": "vidu-q4", "model_type": "video"},
                {"id": "ema/ema-lightning", "object": "model", "owned_by": "ema", "model_type": "tts"},
                {"id": "onysoft/auto", "object": "model", "owned_by": "onysoft", "model_type": "router"},
            ]
        })
    }

    #[test]
    fn chat_model_ids_keeps_only_chat_completions_models() {
        assert_eq!(
            chat_model_ids(&catalog()).unwrap(),
            vec![
                "anthropic/claude-sonnet-5.5".to_string(),
                "google/gemini-3.6-flash".to_string(),
                "x-ai/grok-4.7".to_string(),
            ]
        );
    }

    #[test]
    fn chat_model_ids_rejects_the_non_openai_envelope() {
        // The default `GET /v1/models` body nests the catalog under `data.models`; asking for it
        // without `format=openai` must not silently yield an empty model list.
        let legacy =
            json!({"success": true, "data": {"models": [{"id": "x-ai/grok-4.7"}], "total": 1}});
        assert!(matches!(
            chat_model_ids(&legacy),
            Err(ProviderError::RequestFailed(_))
        ));
    }

    #[test]
    fn chat_model_ids_surfaces_api_errors() {
        let body = json!({"error": {"code": "invalid_api_key", "message": "bad key"}});
        assert!(matches!(
            chat_model_ids(&body),
            Err(ProviderError::Authentication(_))
        ));
    }

    #[test]
    fn credits_errors_carry_the_top_up_url() {
        let err = OnysoftProvider::with_top_up_url(ProviderError::CreditsExhausted {
            details: "out of credit".to_string(),
            top_up_url: None,
        });
        match err {
            ProviderError::CreditsExhausted { top_up_url, .. } => {
                assert_eq!(top_up_url.as_deref(), Some(ONYSOFT_BILLING_URL));
            }
            other => panic!("expected CreditsExhausted, got {other:?}"),
        }
    }

    #[test]
    fn other_errors_are_left_alone() {
        let err = OnysoftProvider::with_top_up_url(ProviderError::Authentication("no".into()));
        assert!(matches!(err, ProviderError::Authentication(_)));
    }
}
