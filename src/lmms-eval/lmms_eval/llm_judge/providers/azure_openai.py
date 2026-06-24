import os
import subprocess
import time
from typing import Dict, Optional

import requests
from loguru import logger as eval_logger

from ..protocol import Request, Response, ServerConfig
from .openai import OpenAIProvider  # Import OpenAIJudge for shared methods


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI implementation of the Judge interface"""

    def __init__(self, config: Optional[ServerConfig] = None):
        super().__init__(config)
        self.api_key = os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.api_endpoint = (
            os.getenv("AZURE_ENDPOINT")
            or os.getenv("AZURE_OPENAI_API_BASE")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("OPENAI_API_BASE")
            or ""
        ).rstrip("/")
        self.api_version = os.getenv("AZURE_API_VERSION") or os.getenv("API_VERSION") or os.getenv("OPENAI_API_VERSION") or "2024-02-15-preview"
        self.azure_ad_token = os.getenv("AZURE_OPENAI_AD_TOKEN", "")
        self.azure_ad_token_provider = None
        if not self.api_key and not self.azure_ad_token:
            try:
                from azure.identity import AzureCliCredential, get_bearer_token_provider

                tenant_id = os.getenv("AZURE_TENANT_ID", "")
                credential = AzureCliCredential(tenant_id=tenant_id) if tenant_id else AzureCliCredential()
                self.azure_ad_token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
            except Exception as e:
                eval_logger.warning(f"Failed to initialize Azure AD token provider: {e}")

        # Fallback: use Azure CLI token when azure-identity provider is unavailable.
        if not self.api_key and not self.azure_ad_token and self.azure_ad_token_provider is None:
            self.azure_ad_token = self._get_cli_token()

        self._azure_client_cls = None
        # Initialize Azure OpenAI client
        try:
            from openai import AzureOpenAI

            self._azure_client_cls = AzureOpenAI
            self.client = self._azure_client_cls(**self._build_client_kwargs())
            self.use_client = True
        except ImportError:
            eval_logger.warning("Azure OpenAI client not available, falling back to requests")
            self.use_client = False
        except Exception as e:
            eval_logger.warning(f"Failed to initialize AzureOpenAI client, falling back to requests: {e}")
            self.use_client = False

    def is_available(self) -> bool:
        has_auth = bool(self.api_key or self.azure_ad_token or self.azure_ad_token_provider is not None)
        return bool(has_auth and self.api_endpoint)

    def evaluate(self, request: Request) -> Response:
        """Evaluate using Azure OpenAI API"""
        if not self.is_available():
            raise ValueError(
                "Azure OpenAI API credentials not configured. "
                "Please set endpoint via AZURE_ENDPOINT (or AZURE_OPENAI_API_BASE) "
                "and auth via AZURE_API_KEY/AZURE_OPENAI_API_KEY, AZURE_OPENAI_AD_TOKEN, "
                "or Azure CLI login (az login)."
            )

        config = request.config or self.config
        messages = self.prepare_messages(request)

        # Handle images if present
        if request.images:
            messages = self._add_images_to_messages(messages, request.images)

        # Prepare payload
        payload = {
            "model": config.model_name,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }

        if config.top_p is not None:
            payload["top_p"] = config.top_p

        if config.response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        # Make API call with retries
        for attempt in range(config.num_retries):
            try:
                if self.use_client:
                    response = self.client.chat.completions.create(**payload)
                    content = response.choices[0].message.content
                    model_used = response.model
                    usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else None
                    raw_response = response
                else:
                    response = self._make_request(payload, config.timeout)
                    content = response["choices"][0]["message"]["content"]
                    model_used = response["model"]
                    usage = response.get("usage")
                    raw_response = response

                return Response(content=content.strip(), model_used=model_used, usage=usage, raw_response=raw_response)

            except Exception as e:
                # Token can expire during long runs; refresh once and retry.
                if self._is_unauthorized_error(e):
                    self._refresh_auth_for_retry()
                eval_logger.warning(f"Attempt {attempt + 1}/{config.num_retries} failed: {str(e)}")
                if attempt < config.num_retries - 1:
                    time.sleep(config.retry_delay)
                else:
                    eval_logger.error(f"All {config.num_retries} attempts failed")
                    raise

    def _make_request(self, payload: Dict, timeout: int) -> Dict:
        """Make HTTP request to Azure OpenAI API"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        else:
            token = self.azure_ad_token
            if not token and self.azure_ad_token_provider is not None:
                token = self.azure_ad_token_provider()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        # Construct the full URL
        deployment_name = payload["model"]
        url = f"{self.api_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version={self.api_version}"

        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def _get_cli_token(self) -> str:
        """Try acquiring Azure AD token via Azure CLI."""
        cmd = [
            "az",
            "account",
            "get-access-token",
            "--resource",
            "https://cognitiveservices.azure.com",
            "--query",
            "accessToken",
            "-o",
            "tsv",
        ]
        tenant_id = os.getenv("AZURE_TENANT_ID", "").strip()
        if tenant_id:
            cmd.extend(["--tenant", tenant_id])

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result.stdout.strip()
        except Exception as e:
            eval_logger.warning(f"Failed to get Azure AD token from Azure CLI: {e}")
            return ""

    def _build_client_kwargs(self) -> Dict:
        client_kwargs = {
            "azure_endpoint": self.api_endpoint,
            "api_version": self.api_version,
        }
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        elif self.azure_ad_token:
            client_kwargs["azure_ad_token"] = self.azure_ad_token
        elif self.azure_ad_token_provider is not None:
            client_kwargs["azure_ad_token_provider"] = self.azure_ad_token_provider
        return client_kwargs

    @staticmethod
    def _is_unauthorized_error(error: Exception) -> bool:
        msg = str(error)
        return "401" in msg or "Unauthorized" in msg

    def _refresh_auth_for_retry(self) -> None:
        # API key users do not need token refresh.
        if self.api_key:
            return

        new_token = self._get_cli_token()
        if not new_token:
            return

        self.azure_ad_token = new_token
        os.environ["AZURE_OPENAI_AD_TOKEN"] = new_token

        if self.use_client and self._azure_client_cls is not None:
            try:
                self.client = self._azure_client_cls(**self._build_client_kwargs())
            except Exception as e:
                eval_logger.warning(f"Failed to rebuild AzureOpenAI client after token refresh: {e}")
