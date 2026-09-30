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

# Kept so older imports don't break; the optimizer it gated was replaced by MCMC.
NEVERGRAD_AVAILABLE = False

try:
    import numpyro  # noqa: F401
    import jax      # noqa: F401
    MCMC_AVAILABLE = True
except ImportError:
    MCMC_AVAILABLE = False
