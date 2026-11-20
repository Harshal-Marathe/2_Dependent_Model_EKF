"""
Optional third-party dependency detection.
Import these flags wherever you need to gate functionality
(Prophet decomposition, holiday calendars, Nevergrad optimizer).
"""

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None
    PROPHET_AVAILABLE = False

try:
    from prophet.make_holidays import make_holidays_df
    HOLIDAYS_AVAILABLE = True
except ImportError:
    make_holidays_df = None
    HOLIDAYS_AVAILABLE = False

try:
    import nevergrad as ng  # noqa: F401
    NEVERGRAD_AVAILABLE = True
except ImportError:
    NEVERGRAD_AVAILABLE = False
