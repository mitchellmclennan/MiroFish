import json
import os
import threading
from typing import Optional
from flask import request, has_request_context

_thread_local = threading.local()

# Run-level LLM output language override. Set to a locale code (e.g. "en")
# to force every LLM prompt site that appends get_language_instruction()
# to emit the matching language directive, independent of request/thread
# locale. Used for bounded runs that must stay in one language end to end
# (ontology, personas, actions/posts episode text, reports).
FORCED_LLM_LANGUAGE_ENV = "MIROFISH_LLM_LANGUAGE"

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
    """Return the run-level forced LLM output language, if any."""

    value = os.environ.get(FORCED_LLM_LANGUAGE_ENV, "").strip().lower()
    return value or None


def is_english_forced() -> bool:
    """Return True when the run is forced to produce English LLM output."""

    return get_forced_llm_language() == "en"


def get_language_instruction() -> str:
    forced = get_forced_llm_language()
    if forced:
        if forced == "en":
            return FORCED_ENGLISH_LLM_INSTRUCTION
        forced_config = _languages.get(forced)
        if forced_config:
            return forced_config.get("llmInstruction", "")
    locale = get_locale()
    lang_config = _languages.get(locale, _languages.get('zh', {}))
    return lang_config.get('llmInstruction', '请使用中文回答。')
