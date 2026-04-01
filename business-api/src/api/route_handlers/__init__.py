"""Business API routes submodules."""

from .analysis_routes import router as analysis_router
from .auth_routes import router as auth_router
from .callback_routes import router as callback_router
from .data_routes import router as data_router
from .deploy_routes import router as deploy_router
from .queue_routes import router as queue_router
from .train_routes import router as train_router

__all__ = [
    "data_router",
    "train_router",
    "deploy_router",
    "analysis_router",
    "auth_router",
    "callback_router",
    "queue_router",
]
