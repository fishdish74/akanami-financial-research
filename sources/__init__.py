"""Discovery source adapters used by the V1 collector."""

from .google_news import GoogleNewsRssSource
from .manual import ManualJsonSource
from .sec import SecSubmissionsSource

__all__ = ["GoogleNewsRssSource", "ManualJsonSource", "SecSubmissionsSource"]
