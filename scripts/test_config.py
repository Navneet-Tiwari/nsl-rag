"""
test_config.py
--------------
Quick sanity check for config and logger setup.
Run: python scripts/test_config.py
"""

from nsl_rag.core.logger import setup_logging, get_logger
from nsl_rag.config.config_loader import config

setup_logging()
log = get_logger("nsl_rag.test")

config.load()

log.info("model          : %s", config.llm.model)
log.info("max_depth      : %s", config.lattice.max_depth)
log.info("environment    : %s", config.settings.project.environment)
log.info("config OK")
