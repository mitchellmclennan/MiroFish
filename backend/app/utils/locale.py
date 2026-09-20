import json
import os
import threading
from typing import Optional
from flask import request, has_request_context

from .logger import get_logger

logger = get_logger('mirofish.locale')

_thread_local = threading.local()

# Run-level LLM output language override. Set to a locale code (e.g. "en")
# to force every LLM prompt site that appends get_language_instruction()
# to emit the matching language directive, independent of request/thread
# locale. Used for bounded runs that must stay in one language end to end
# (ontology, personas, actions/posts episode text, reports).
#
# Only "en" is actually supported end to end: the English persona/config
# templates exist solely as English variants, so forcing any other language
# would produce mixed-language output. Region variants (en-US, en_US) are
# normalized to "en"; other values are rejected with a warning and the run
# falls back to locale-driven behavior.
FORCED_LLM_LANGUAGE_ENV = "MIROFISH_LLM_LANGUAGE"

# Only this primary language has full pipeline support for run-level forcing.
SUPPORTED_FORCED_LLM_LANGUAGE = "en"

# Warn once per unsupported value (get_forced_llm_language is called at
# every prompt site, so per-call logging would be too noisy).
_unsupported_language_warnings: set = set()

# English must win even when a surrounding Chinese prompt template asks
# for Chinese field values, so the forced variant is stronger than the
# generic per-locale instruction below.
FORCED_ENGLISH_LLM_INSTRUCTION = (
    "Please respond in English only. All natural-language text you produce, "
    "including every JSON string value, MUST be written in English. "
    "Do not write any Chinese or other non-English text in your output."
)

_locales_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'locales')

# Load language registry
with open(os.path.join(_locales_dir, 'languages.json'), 'r', encoding='utf-8') as f:
    _languages = json.load(f)

# Load translation files
_translations = {}
for filename in os.listdir(_locales_dir):
    if filename.endswith('.json') and filename != 'languages.json':
        locale_name = filename[:-5]
        with open(os.path.join(_locales_dir, filename), 'r', encoding='utf-8') as f:
            _translations[locale_name] = json.load(f)


def set_locale(locale: str):
    """Set locale for current thread. Call at the start of background threads."""
    _thread_local.locale = locale


def get_locale() -> str:
    if has_request_context():
        raw = request.headers.get('Accept-Language', 'zh')
        return raw if raw in _translations else 'zh'
    return getattr(_thread_local, 'locale', 'zh')


def t(key: str, **kwargs) -> str:
    locale = get_locale()
    messages = _translations.get(locale, _translations.get('zh', {}))

    value = messages
    for part in key.split('.'):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            value = None
            break

    if value is None:
        value = _translations.get('zh', {})
        for part in key.split('.'):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break

    if value is None:
        return key

    if kwargs:
        for k, v in kwargs.items():
            value = value.replace(f'{{{k}}}', str(v))

    return value


def get_forced_llm_language() -> Optional[str]:
    """Return the run-level forced LLM output language, if any.

    Region variants of the supported language are normalized to the
    primary code ("en-US"/"en_US" -> "en"). Any other language is not
    supported for run-level forcing (only English has full-pipeline
    templates); it is rejected with a one-time warning and the run falls
    back to locale-driven behavior instead of producing mixed-language
    output.
    """
    raw = os.environ.get(FORCED_LLM_LANGUAGE_ENV, "").strip()
    if not raw:
        return None
    primary = raw.replace("_", "-").split("-")[0].strip().lower()
    if primary == SUPPORTED_FORCED_LLM_LANGUAGE:
        return SUPPORTED_FORCED_LLM_LANGUAGE
    if primary not in _unsupported_language_warnings:
        _unsupported_language_warnings.add(primary)
        logger.warning(
            f"{FORCED_LLM_LANGUAGE_ENV}='{raw}' 不受支持：目前仅支持强制英文 'en'"
            f"（区域变体如 en-US 会被归一化为 en）。其他语言没有完整的模板"
            f"支持，强行生效会产生混合语言输出；已忽略该设置并回退到locale行为"
        )
    return None


def is_english_forced() -> bool:
    """Return True when the run is forced to produce English LLM output."""

    return get_forced_llm_language() == "en"


def get_language_instruction() -> str:
    forced = get_forced_llm_language()
    if forced:
        # Run-level forcing only supports English (region variants are
        # normalized inside get_forced_llm_language; other languages are
        # rejected there with a warning and fall through to locale).
        if forced == "en":
            return FORCED_ENGLISH_LLM_INSTRUCTION
    locale = get_locale()
    lang_config = _languages.get(locale, _languages.get('zh', {}))
    return lang_config.get('llmInstruction', '请使用中文回答。')
