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
    """Validate API key via Authorization Bearer token or X-API-Key header against valid keys."""
    token_candidate = None
    if bearer and bearer.credentials:
        token_candidate = bearer.credentials.strip()
    elif x_api_key:
        token_candidate = x_api_key.strip()

    is_valid = False
    if token_candidate:
        for valid_key in settings.valid_api_keys:
            if secrets.compare_digest(token_candidate, valid_key):
                is_valid = True
                break

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_candidate


