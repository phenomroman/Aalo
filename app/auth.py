import secrets
from typing import Optional
from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.config import Settings, get_settings

security_bearer = HTTPBearer(auto_error=False)


def verify_api_key(
    bearer: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> str:
    """Validate API key via Authorization Bearer token or X-API-Key header."""
    token_candidate = None
    if bearer and bearer.credentials:
        token_candidate = bearer.credentials
    elif x_api_key:
        token_candidate = x_api_key

    # Compare using constant-time comparison to prevent timing attacks
    if not token_candidate or not secrets.compare_digest(token_candidate, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_candidate

