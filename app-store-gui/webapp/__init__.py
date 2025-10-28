# Make webapp a proper Python package so `uvicorn webapp.main:app --reload` works reliably.
# Also export `app` at the package level for convenience: `uvicorn webapp:app --reload`.
from .main import app as app
