import structlog
import logging
from pathlib import Path

def setup_logger(log_dir: str = "data/logs", log_level: str = "INFO"):
    """
    Sets up structured logging for the VLM Danger System.
    Outputs to both console (colored) and a JSON file.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Standard library logging setup
    logging.basicConfig(
        format="%(message)s",
        stream=open(f"{log_dir}/system.log", "a"),
        level=level,
    )
    
    # Also log to console for debugging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(console_handler)

    return structlog.get_logger()

# Create a default logger instance
logger = structlog.get_logger()
