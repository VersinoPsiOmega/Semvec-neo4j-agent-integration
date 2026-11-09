"""semvec_neo4j — Semvec × Neo4j integration.

Drift detection, multi-agent clusters, regions/observer, and the INVESTIGATED
bridge into a healthcare knowledge graph.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("semvec_neo4j")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
