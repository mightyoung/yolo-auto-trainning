"""Training API route handlers package.

This package contains route handlers extracted from routes.py.
The monolithic routes.py was renamed to _routes_impl.py, and the remaining
deployment / continuous-training endpoints now live in sibling modules.

Structure:
- __init__.py: Re-exports router from _routes_impl
- _routes_impl.py: Core route implementations
- continuous.py: Continuous-training pipeline routes
- deploy.py: Drift detection and edge deployment routes

Migration status:
- models/ - DONE
- store/ - DONE
- services/ - DONE (_run_*_sync functions in services/_shared.py)
- routes/route_modules - IN PROGRESS
"""

from ._routes_impl import router

__all__ = ["router"]
