"""Training API route handlers package.

This package contains route handlers extracted from routes.py.
The monolithic routes.py was renamed to _routes_impl.py.

Structure:
- __init__.py: Re-exports router from _routes_impl
- _routes_impl.py: Original route implementations (renamed from routes.py)
- _shared.py: Extracted shared utilities (_run_*_sync functions, DynamicTrainingManager)

Migration status:
- models/ - DONE
- store/ - DONE
- services/ - DONE
- routes/_shared.py - DONE
- routes/route_modules - TODO (individual route files)
"""

from ._routes_impl import router

__all__ = ["router"]
