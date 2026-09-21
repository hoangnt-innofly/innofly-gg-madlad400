from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.core.config import get_settings

api_key_header = APIKeyHeader(name="api-key", auto_error=False)


def require_api_key(api_key: Annotated[str | None, Depends(api_key_header)]) -> str:
    expected = (get_settings().secret_api_key or "").strip()
    provided = (api_key or "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SECRET_API_KEY is not configured",
        )
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing api-key header",
        )
    return provided
