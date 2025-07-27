"""Admin API endpoints"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get("/stats")
async def get_admin_stats(db: Session = Depends(get_db)):
    """Get admin statistics."""
    return {"message": "Admin stats endpoint", "status": "ok"}


@router.post("/users/roles")
async def assign_user_role(role_data: dict, db: Session = Depends(get_db)):
    """Assign role to user."""
    # Mock implementation for testing
    return {"message": "Role assigned successfully"}


@router.get("/team")
async def get_team_info(db: Session = Depends(get_db)):
    """Get team information."""
    # Mock implementation for testing
    return {"team": "Manager team", "members": []}
