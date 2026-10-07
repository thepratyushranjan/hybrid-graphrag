"""Thin HTTP client the Streamlit UI uses to talk to the FastAPI service."""

from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class ApiResult:
    ok: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    not_implemented: bool = False


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 900.0) -> None:  # ingest runs LLM extraction
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> ApiResult:
        try:
            resp = httpx.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        except httpx.HTTPError as exc:
            return ApiResult(ok=False, error=f"Cannot reach API at {self.base_url}: {exc}")

        if resp.status_code in (404, 405):
            return ApiResult(ok=False, not_implemented=True, error=f"{method} {path} is not implemented yet")
        try:
            body = resp.json()
        except ValueError:
            body = {"detail": resp.text}
        if resp.is_success:
            return ApiResult(ok=True, data=body)
        # /health returns a useful body with 503, so keep it
        return ApiResult(ok=False, data=body, error=str(body.get("detail", f"HTTP {resp.status_code}")))

    def health(self) -> ApiResult:
        return self._request("GET", "/health")

    def ingest(self, filename: str, content: bytes, content_type: str) -> ApiResult:
        return self._request("POST", "/ingest", files={"file": (filename, content, content_type)})

    def query(self, question: str, top_k: int, hops: int, mode: str = "hybrid") -> ApiResult:
        return self._request(
            "POST", "/query", json={"question": question, "top_k": top_k, "hops": hops, "mode": mode}
        )
