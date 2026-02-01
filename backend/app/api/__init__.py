"""API routes for RigaCap."""

from fastapi import APIRouter

from app.api.signals import router as signals_router
from app.api.auth import router as auth_router
from app.api.billing import router as billing_router
from app.api.admin import router as admin_router

api_router = APIRouter()

api_router.include_router(signals_router, prefix="/signals", tags=["signals"])
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(billing_router, prefix="/billing", tags=["billing"])
api_router.include_router(admin_router, prefix="/admin", tags=["admin"])
