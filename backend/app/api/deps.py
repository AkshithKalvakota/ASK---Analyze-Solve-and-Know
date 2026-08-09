from fastapi import Request, HTTPException
from clerk_backend_api import Clerk
from clerk_backend_api.security.types import AuthenticateRequestOptions
from app.core.config import settings

clerk = Clerk(bearer_auth=settings.clerk_secret_key)

def get_current_user_id(request: Request) -> str:
    try:
        request_state = clerk.authenticate_request(
            request,
            AuthenticateRequestOptions(
                authorized_parties=[
                    "http://localhost:5173",
                    "https://ask-analyze-solve-and-know.vercel.app",
                ],
            ),
        )
        if not request_state.is_signed_in:
            raise HTTPException(status_code=401, detail="Not signed in")
        return request_state.payload["sub"]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Auth error: {str(e)}")