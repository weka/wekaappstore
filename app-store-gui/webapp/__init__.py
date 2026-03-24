# Make webapp a proper Python package so `uvicorn webapp.main:app --reload` works reliably.
# Also export `app` at the package level for convenience: `uvicorn webapp:app --reload`.
#
# Lazy import: importing webapp as a library (e.g., from mcp-server tools) does NOT
# pull in FastAPI and all GUI dependencies. Only import `app` when the GUI is the target.
# Use `uvicorn webapp.main:app` or `from webapp.main import app` for the GUI server.
def __getattr__(name: str):
    if name == "app":
        from .main import app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
