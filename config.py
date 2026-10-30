"""Central, validated configuration for the manually triggered V1 collector."""

from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


PROJECT_DIR = Path(__file__).resolve().parent
VAULT_DIR = PROJECT_DIR.parent.parent
ENV_PATH = PROJECT_DIR / ".env"

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_env_value(raw: str) -> str:
    """Parse the small KEY=value subset used by this project without executing it."""

    value = raw.strip()
    if value.startswith(("'", '"')):
        quote = value[0]
        closing = value.find(quote, 1)
        if closing < 0:
            raise ValueError("environment value contains an unmatched quote")
        remainder = value[closing + 1 :].strip()
        if remainder and not remainder.startswith("#"):
            raise ValueError("unexpected text after quoted environment value")
        return value[1:closing]
    if "'" in value or '"' in value:
        raise ValueError("environment value contains an unmatched quote")
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def load_env_file(path: Path = ENV_PATH) -> set[str]:
    """Load an optional local .env file; existing process variables always win.

    The return value contains only variable names, never values, so callers can
    report configuration safely.
    """

    loaded: set[str] = set()
    if not path.is_file():
        return loaded
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid .env entry on line {line_number}")
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f"invalid .env variable name on line {line_number}")
        os.environ.setdefault(name, _parse_env_value(raw_value))
        loaded.add(name)
    return loaded


LOADED_ENV_NAMES = load_env_file()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def writes_enabled() -> bool:
    """Return true only for the single explicit opt-in value."""

    return os.getenv("MARKET_COLLECTOR_ENABLE_WRITES", "0") == "1"


def api_key_is_configured() -> bool:
    """Report key presence without exposing its value."""

    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def require_sec_user_agent(value: str) -> str:
    """Validate the SEC-required identifying application/contact string once."""

    user_agent = value.strip()
    email_match = re.search(
        r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+",
        user_agent,
    )
    contact_span = email_match.span() if email_match else None
    if contact_span is None:
        for match in re.finditer(r"https?://[^\s]+", user_agent, flags=re.IGNORECASE):
            candidate = match.group().rstrip(".,;:)]}")
            parsed = urlsplit(candidate)
            if parsed.scheme.casefold() in {"http", "https"} and parsed.hostname:
                contact_span = match.span()
                break
    application = user_agent
    if contact_span is not None:
        application = (user_agent[: contact_span[0]] + user_agent[contact_span[1] :]).strip()
    identifiable_application = len(re.findall(r"[A-Za-z0-9]", application)) >= 2
    if not user_agent or contact_span is None or not identifiable_application:
        raise ValueError(
            "SEC_USER_AGENT must identify the application and include a contact email or URL"
        )
    return user_agent


@dataclass(frozen=True)
class SourceRequestConfig:
    delay_seconds: float
    timeout_seconds: float
    retries: int
    headers: dict[str, str]

    def __post_init__(self) -> None:
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds cannot be negative")
        if self.timeout_seconds < 0:
            raise ValueError("timeout_seconds cannot be negative")
        if self.retries < 0:
            raise ValueError("retries cannot be negative")


@dataclass(frozen=True)
class ModelConfig:
    model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-6-luna"))
    prompt_version: str = "market-analysis-v7"
    pipeline_version: str = "v1"
    input_cost_per_million: float = field(
        default_factory=lambda: _env_float("OPENAI_INPUT_COST_PER_MILLION", 0.10)
    )
    output_cost_per_million: float = field(
        default_factory=lambda: _env_float("OPENAI_OUTPUT_COST_PER_MILLION", 0.50)
    )
    max_input_chars: int = field(
        default_factory=lambda: _env_int("MAX_ANALYSIS_INPUT_CHARS", 50_000)
    )
    max_output_tokens: int = field(
        default_factory=lambda: _env_int("OPENAI_MAX_OUTPUT_TOKENS", 4_000)
    )
    timeout_seconds: float = field(
        default_factory=lambda: _env_float("OPENAI_TIMEOUT_SECONDS", 30.0)
    )

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("OPENAI_MODEL cannot be empty")
        if not all(math.isfinite(value) and value >= 0 for value in (
            self.input_cost_per_million, self.output_cost_per_million
        )):
            raise ValueError("model costs cannot be negative")
        if self.max_input_chars <= 0:
            raise ValueError("MAX_ANALYSIS_INPUT_CHARS must be positive")
        if self.max_output_tokens <= 0:
            raise ValueError("OPENAI_MAX_OUTPUT_TOKENS must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("OPENAI_TIMEOUT_SECONDS must be a positive finite number")


@dataclass(frozen=True)
class CollectorConfig:
    vault_dir: Path = VAULT_DIR
    article_dir: Path = VAULT_DIR / "03 Articles"
    registry_path: Path = PROJECT_DIR / "entities.json"
    max_deep_calls: int = field(
        default_factory=lambda: _env_int("MAX_DEEP_ANALYSIS_CALLS", 3)
    )
    max_ai_budget_usd: float = field(
        default_factory=lambda: _env_float("MAX_AI_BUDGET_USD", 0.25)
    )
    min_content_chars: int = field(
        default_factory=lambda: _env_int("MIN_CONTENT_CHARS", 80)
    )
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        if self.max_deep_calls <= 0:
            raise ValueError("MAX_DEEP_ANALYSIS_CALLS must be positive")
        if not math.isfinite(self.max_ai_budget_usd) or self.max_ai_budget_usd < 0:
            raise ValueError("MAX_AI_BUDGET_USD must be a non-negative finite number")
        if self.min_content_chars < 0:
            raise ValueError("MIN_CONTENT_CHARS cannot be negative")


def source_configs() -> dict[str, SourceRequestConfig]:
    sec_user_agent = os.getenv("SEC_USER_AGENT", "")
    return {
        "sec": SourceRequestConfig(
            delay_seconds=0.12,
            timeout_seconds=15.0,
            retries=2,
            headers={"User-Agent": sec_user_agent, "Accept-Encoding": "gzip, deflate"},
        ),
        "google-news": SourceRequestConfig(
            delay_seconds=1.0,
            timeout_seconds=15.0,
            retries=2,
            headers={"User-Agent": "MarketResearchCollector/0.2 (manual prototype)"},
        ),
        "manual": SourceRequestConfig(
            delay_seconds=0.0,
            timeout_seconds=0.0,
            retries=0,
            headers={},
        ),
    }
