"""Dashboard package for web-based monitoring.

Provides a Flask-based dashboard for monitoring trading activity.
"""

try:
    from .app import DashboardApp, create_app
    
    __all__ = [
        "DashboardApp",
        "create_app",
    ]
except ImportError:
    # Flask not available
    __all__ = []
