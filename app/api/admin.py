"""Admin API endpoints"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get("/stats")
async def get_admin_stats(db: Session = Depends(get_db)):
    """Get admin statistics."""
    return {"message": "Admin stats endpoint", "status": "ok"}
