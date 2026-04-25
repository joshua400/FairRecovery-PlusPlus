"""FairRecovery++ — Logging Configuration."""

from __future__ import annotations
import logging, sys


def configure_logging(json_output: bool = True, log_level: str = "INFO") -> None:
    try:
        import structlog
        shared = [structlog.contextvars.merge_contextvars, structlog.stdlib.add_log_level,
                  structlog.stdlib.add_logger_name, structlog.processors.TimeStamper(fmt="iso")]
        renderer = structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
        structlog.configure(
            processors=shared + [structlog.stdlib.PositionalArgumentsFormatter(),
                                 structlog.processors.StackInfoRenderer(),
                                 structlog.processors.format_exc_info, renderer],
            wrapper_class=structlog.stdlib.BoundLogger, context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(), cache_logger_on_first_use=True)
    except ImportError:
        pass
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stderr, level=getattr(logging, log_level.upper(), logging.INFO))
