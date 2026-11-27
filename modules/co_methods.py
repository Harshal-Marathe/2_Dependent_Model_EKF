"""Which Cochrane-Orcutt method(s) a config asks for.

"ar1"      classical CO: estimate rho from the residuals, quasi-difference the
           observation equation, refit, repeat (modules/cochrane_orcutt.py).
"residual" fit, add the residuals as a variable, re-estimate everything
           (modules/residual_refit.py).

Either or both may be selected (config["co_methods"]). Configs saved before
the multi-select existed carry the single config["co_method"] (default "ar1").
"""

METHODS = ("ar1", "residual")


def co_methods(cfg):
    if not cfg or not cfg.get("use_cochrane_orcutt"):
        return []
    m = cfg.get("co_methods")
    if m is None:
        m = [cfg.get("co_method", "ar1")]
    return [x for x in METHODS if x in m]
