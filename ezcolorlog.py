"""
Minimal local fallback for environments where the external ezcolorlog package
pulls in optional notebook/runtime dependencies that are unavailable.
"""

import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

root_logger = logging.getLogger("scale_rae")
