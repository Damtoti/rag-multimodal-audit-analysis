"""Compatibility package initializer for mixed src-layout deployments."""
from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_src_pkg = Path(__file__).resolve().parent / "src" / "audit_rag"
if _src_pkg.exists():
    src_pkg_str = str(_src_pkg)
    if src_pkg_str not in __path__:
        __path__.append(src_pkg_str)
