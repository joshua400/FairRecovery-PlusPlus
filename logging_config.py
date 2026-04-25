"""
FairRecovery — Logging Configuration.

Provides structured JSON logging via structlog.
Mirrors hallucination-detector-gym logging_config.py.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(
    json_output: bool = True,
    log_level: str = "INFO",
) -> None:
    """Configure structlog for structured JSON output.

    Args:
        json_output: If True, emit JSON lines. If False, emit human-readable.
        log_level:   Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    try:
        import structlog

        shared_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
        ]

        if json_output:
            renderer = structlog.processors.JSONRenderer()
        else:
            renderer = structlog.dev.ConsoleRenderer()  # type: ignore

        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                renderer,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    except ImportError:
        # structlog not available — fall back to stdlib
        pass

    # Always configure stdlib root logger
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
