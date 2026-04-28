"""LLM client with graceful provider fallback.

Provider chain (first available wins, with fall-through on errors/limits):
  1. Groq cloud inference (configurable models via GROQ_MODELS)
  2. Local Ollama (configurable model via OLLAMA_MODEL)
  3. Deterministic rule-based engine — always available offline.

If GROQ_API_KEY is unset, the chain skips straight to Ollama. If neither
LLM is reachable, the rule-based engine still produces classified output
so the dashboard works on any environment.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT_S = float(os.environ.get("OLLAMA_TIMEOUT_S", "30"))

# --- Groq (cloud, fast inference for open-source models) ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
# Model fallback chain inside Groq itself: try the larger first, fall back
# to a smaller/faster one if the first hits a rate limit.
GROQ_MODELS = [m.strip() for m in os.environ.get(
    "GROQ_MODELS",
    "llama-3.3-70b-versatile,llama-3.1-8b-instant,gemma2-9b-it"
).split(",") if m.strip()]
GROQ_TIMEOUT_S = float(os.environ.get("GROQ_TIMEOUT_S", "30"))


# ---------- runtime probe ----------

@lru_cache(maxsize=1)
def ollama_available() -> bool:
    """Probe Ollama once per process. Cached so we don't spam the VM."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if r.status_code != 200:
            return False
        tags = r.json().get("models", [])
        names = {m.get("name", "") for m in tags}
        base = OLLAMA_MODEL.split(":")[0]
        return OLLAMA_MODEL in names or any(n.startswith(base) for n in names)
    except Exception as exc:  # pragma: no cover
        logger.debug("Ollama probe failed: %s", exc)
        return False


def groq_available() -> bool:
    """Groq is available iff an API key is configured. We don't probe network
    proactively — a single failed request triggers our fallback path."""
    return bool(GROQ_API_KEY)


def llm_available() -> bool:
    """True if ANY LLM provider is reachable. Used by the dashboard to light
    up free-form Q&A and similar features."""
    return groq_available() or ollama_available()


def runtime_status() -> dict[str, Any]:
    """Frontend uses this to display the active provider chain."""
    return {
        "ollama_url": OLLAMA_BASE_URL,
        "ollama_model": OLLAMA_MODEL,
        "ollama_available": ollama_available(),
        "groq_available": groq_available(),
        "groq_models": GROQ_MODELS,
        "llm_available": groq_available() or ollama_available(),
        # backward-compat alias used by older clients
        "active_provider": (
            "groq" if groq_available() else
            "ollama" if ollama_available() else
            "rule"
        ),
    }


# ---------- low-level calls ----------

class _RateLimited(Exception):
    """Raised when a provider returns 429 so the chain skips to the next."""


def _groq_generate(prompt: str, system: str | None = None,
                   temperature: float = 0.2,
                   model: str | None = None) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    payload = {
        "model": model or GROQ_MODELS[0],
        "messages": msgs,
        "temperature": temperature,
        "stream": False,
    }
    r = requests.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=GROQ_TIMEOUT_S,
    )
    if r.status_code == 429:
        raise _RateLimited(f"Groq rate-limited (model={payload['model']})")
    if r.status_code >= 400:
        raise RuntimeError(f"Groq HTTP {r.status_code}: {r.text[:200]}")
    data = r.json()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()


def _ollama_generate(prompt: str, system: str | None = None,
                     temperature: float = 0.2) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=OLLAMA_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _llm_generate(prompt: str, system: str | None = None,
                  temperature: float = 0.2) -> tuple[str, str]:
    """Generate text using the provider chain.

    Order:
      1. Groq with each model in GROQ_MODELS, falling through on rate-limits.
      2. Ollama with OLLAMA_MODEL.
    Returns (text, provider_used). Raises if all providers fail.
    """
    last_exc: Exception | None = None

    # 1) Groq cascade
    if groq_available():
        for model in GROQ_MODELS:
            try:
                text = _groq_generate(prompt, system=system,
                                       temperature=temperature, model=model)
                if text:
                    return text, f"groq:{model}"
            except _RateLimited as exc:
                logger.info("Groq model rate-limited, trying next: %s", exc)
                last_exc = exc
                continue
            except Exception as exc:  # pragma: no cover
                logger.warning("Groq error on %s: %s", model, exc)
                last_exc = exc
                continue

    # 2) Ollama
    if ollama_available():
        try:
            text = _ollama_generate(prompt, system=system, temperature=temperature)
            if text:
                return text, f"ollama:{OLLAMA_MODEL}"
        except Exception as exc:
            logger.warning("Ollama error: %s", exc)
            last_exc = exc

    if last_exc:
        raise last_exc
    raise RuntimeError("no LLM provider available")


def _try_parse_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM response."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ---------- public API ----------

@dataclass
class Enrichment:
    severity: str           # "عالية" | "متوسطة" | "منخفضة"
    severity_reason: str    # one-sentence Arabic justification
    topic_label: str        # short Arabic label
    recommended_action: str # one-line next step in Arabic
    source: str             # "llm" | "rule"


SEVERITY_VALUES = ("عالية", "متوسطة", "منخفضة")


# --- Rule-based engine ---
# The patterns below are tuned to financial-services customer support:
# loans/financing, repayment & collections, beneficiary subsidies, and
# digital channels. They produce severity, topic, and recommended action
# for each request without requiring any LLM.

# Patterns: phrase fragments → (severity, topic_hint, business interpretation)
# Order matters: more specific patterns first.
_HIGH_PATTERNS = [
    (("متعسر", "تعسر", "تعثر"),
     "تعثّر في السداد",
     "إشارة إلى حالة تعثّر في السداد، وهي من أعلى المخاطر التشغيلية لدى الصندوق وتستوجب تواصلاً مع لجنة التحصيل."),
    (("احتيال", "تزوير", "تلاعب"),
     "بلاغ احتيال",
     "إشارة محتملة إلى احتيال أو تلاعب، تتطلب تصعيداً فورياً لإدارة المخاطر والامتثال."),
    (("خصم خاطئ", "خصم زائد", "حُسم", "حسم خاطئ"),
     "خصم مالي مُتنازَع عليه",
     "ادعاء بخصم مالي غير صحيح من حساب المستفيد، يستلزم مراجعة فورية لفك الإشكال."),
    (("ايقاف الحسم", "إيقاف الحسم", "إيقاف الخصم", "ايقاف الخصم"),
     "إيقاف الخصم على القرض",
     "طلب إيقاف الخصم على قرض قائم — قرار مالي حساس يؤثر على جدولة السداد."),
    (("متأخر", "تأخر", "تأخير", "لم يُصرف", "لم يصرف", "بدون رد", "بلا رد"),
     "تأخر في الإجراء",
     "ادعاء بتأخر في معالجة طلب أو صرف مالي، يؤثر مباشرةً على المستفيد ويتطلب متابعة عاجلة."),
    (("رفض القرض", "رفض الطلب", "تم رفض"),
     "رفض طلب",
     "اعتراض على رفض طلب تمويل، يتطلب توضيحاً لأسباب القرار وفق سياسة الصندوق."),
    (("استرداد", "استرجاع المبلغ"),
     "طلب استرداد",
     "طلب استرداد مبلغ مالي، يتطلب التحقق من سجل المعاملة وإجراء التسوية."),
    (("عاجل", "مستعجل", "ضروري"),
     "طلب عاجل",
     "وُصف الطلب بأنه عاجل من قِبل المستفيد، يستحق التعامل ضمن مسار الأولوية."),
]

_MEDIUM_PATTERNS = [
    (("سداد", "تسديد", "قسط", "أقساط"),
     "السداد والأقساط",
     "استعلام أو طلب يخصّ جدولة السداد أو الأقساط، يستحق متابعة لتفادي التحوّل إلى تعثّر."),
    (("قرض", "تمويل"),
     "القروض والتمويل",
     "طلب يخصّ منتج تمويلي للصندوق، يتطلب توضيحاً من فريق الائتمان."),
    (("دعم زراعي", "دعم المزارعين", "دعم الإنتاج"),
     "الدعم الزراعي",
     "طلب يخصّ برامج الدعم الزراعي، يستحق إحالة إلى الجهة المختصة."),
    (("دواجن", "ماشية", "أبقار", "أغنام", "نحل", "أسماك"),
     "الإنتاج الحيواني",
     "طلب يخصّ نشاط إنتاج حيواني ضمن نطاق الصندوق."),
    (("مشروع", "مزرعة"),
     "مشاريع المستفيدين",
     "طلب يخصّ مشروعاً قائماً للمستفيد، يستحق متابعة للحفاظ على استمراريته."),
    (("تحديث بيانات", "تحديث رقم", "تحديث الجوال"),
     "تحديث البيانات",
     "طلب تحديث بيانات تواصل أو شخصية، إجراء روتيني لكن يلزم لإكمال الخدمات."),
    (("بوابة", "تطبيق", "موقع", "نظام"),
     "القنوات الرقمية",
     "طلب يخصّ تجربة الاستخدام في القنوات الرقمية للصندوق."),
]

_LOW_PATTERNS = [
    (("شكر", "تقدير", "ممتاز"),
     "إشادة وثناء",
     "إشادة من المستفيد، لا تستوجب إجراءً تصحيحياً."),
    (("اقتراح", "مقترح", "تطوير"),
     "اقتراح تطوير",
     "اقتراح تحسين من المستفيد، يُحوَّل إلى سجل التحسينات."),
    (("استعلام", "استفسار", "كيف", "متى", "ما هي"),
     "استعلام عام",
     "طلب معلومة عامة، يمكن الرد عليه من قاعدة المعرفة دون تصعيد."),
]


def _excerpt(body: str, limit: int = 60) -> str:
    """Short, clean excerpt of the body for use in justifications."""
    s = (body or "").strip()
    if not s:
        return ""
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit].rstrip() + "…"


def _match_pattern(text: str, patterns: list) -> tuple | None:
    for keys, topic, reason in patterns:
        for k in keys:
            if k in text:
                return (k, topic, reason)
    return None


def _rule_classify(category: str, body: str) -> tuple[str, str, str]:
    """Return (severity, topic_label, severity_reason)."""
    cat = (category or "").strip()
    body_clean = (body or "").strip()
    text = f"{cat} {body_clean}"
    excerpt = _excerpt(body_clean)

    # 1) High-severity domain-specific signals
    hit = _match_pattern(text, _HIGH_PATTERNS)
    if hit:
        keyword, topic, reason = hit
        if excerpt and excerpt != keyword:
            reason += f" نص الطلب: «{excerpt}»."
        return "عالية", topic, reason

    # 2) Category-driven default for complaints
    if cat == "شكوى":
        if excerpt:
            return ("عالية", "الشكاوى العامة",
                    f"شكوى مباشرة من المستفيد بشأن «{excerpt}» — تتطلب تصعيداً وتواصلاً استباقياً للحفاظ على رضا المتعامل.")
        return ("عالية", "الشكاوى العامة",
                "شكوى مباشرة بدون وصف تفصيلي — تستدعي اتصالاً استباقياً لاستيضاح المشكلة.")

    # 3) Medium signals
    hit = _match_pattern(text, _MEDIUM_PATTERNS)
    if hit:
        keyword, topic, reason = hit
        if excerpt:
            reason += f" نص الطلب: «{excerpt}»."
        return "متوسطة", topic, reason

    if cat == "دعم فني":
        return ("متوسطة", "الدعم الفني للقنوات الرقمية",
                f"طلب دعم فني{(' بشأن «' + excerpt + '»') if excerpt else ''} — يؤثر على قدرة المستفيد على إكمال خدمته.")

    if cat == "خدمة مراجع":
        return ("متوسطة", "خدمة المراجعين",
                f"طلب يخصّ مسار خدمة المراجعين{(' حول «' + excerpt + '»') if excerpt else ''} — يتطلب متابعة لاستكمال إجراءات المستفيد.")

    # 4) Low signals
    hit = _match_pattern(text, _LOW_PATTERNS)
    if hit:
        keyword, topic, reason = hit
        if excerpt:
            reason += f" نص الطلب: «{excerpt}»."
        return "منخفضة", topic, reason

    if cat in ("استفسار", "اقتراح"):
        topic = "استعلام عام" if cat == "استفسار" else "اقتراح تطوير"
        reason = (
            f"طلب من نوع «{cat}»{(' حول «' + excerpt + '»') if excerpt else ''} — "
            "لا يستوجب تصعيداً ويمكن الرد ضمن المسار الاعتيادي."
        )
        return "منخفضة", topic, reason

    # 5) Default
    return ("متوسطة", cat or "غير مصنّف",
            f"تصنيف افتراضي{(' للطلب بشأن «' + excerpt + '»') if excerpt else ''} — لم تُكتشف إشارات صريحة تحدد مستوى الأولوية.")


def _rule_topic(category: str, body: str) -> str:
    return _rule_classify(category, body)[1]


_ACTION_BY_CATEGORY_AND_SEVERITY = {
    ("شكوى", "عالية"): "تصعيد فوري إلى مختص الشكاوى — اتصال خلال 24 ساعة + تسجيل في نظام الجودة",
    ("شكوى", "متوسطة"): "إحالة إلى فريق رعاية المستفيدين مع SLA 48 ساعة",
    ("دعم فني", "عالية"): "تذكرة دعم فوري + التحقق من توفّر الخدمة الرقمية على مستوى الصندوق",
    ("دعم فني", "متوسطة"): "فتح تذكرة دعم فني والرد بحلّ موثّق",
    ("استفسار", "منخفضة"): "الرد من قاعدة المعرفة وإغلاق الطلب مع تأكيد الفهم",
    ("استفسار", "متوسطة"): "تحويل إلى المختص للتأكد من المعلومة قبل الرد",
    ("اقتراح", "منخفضة"): "تسجيل المقترح في سجل التحسينات وإحالته إلى لجنة التطوير",
    ("خدمة مراجع", "متوسطة"): "تحويل إلى مسار خدمة المراجعين وإبلاغ المستفيد بالمتطلبات الناقصة",
    ("خدمة مراجع", "عالية"): "خدمة مراجعين عاجلة — متابعة المستفيد ومراجعة الملف",
}


def _rule_action(category: str, body: str, severity: str, topic: str) -> str:
    cat = (category or "").strip()
    base = _ACTION_BY_CATEGORY_AND_SEVERITY.get((cat, severity))
    if base:
        return base
    # Topic-driven fallbacks for high severity
    if severity == "عالية":
        if "تعثّر" in topic or "تعثر" in topic:
            return "إحالة فورية إلى لجنة التحصيل وتفعيل خطة استرداد مرنة وفق وضع المستفيد"
        if "احتيال" in topic:
            return "تصعيد إلى إدارة المخاطر والامتثال خلال ساعة + إيقاف العمليات المرتبطة"
        if "خصم" in topic:
            return "مراجعة سجل الخصم خلال 24 ساعة وإجراء التسوية المالية إن ثبت الخطأ"
        if "تأخر" in topic:
            return "متابعة الطلب لدى الجهة المختصة واتصال استباقي بالمستفيد بنتيجة المراجعة"
    if severity == "متوسطة":
        return "إحالة إلى الجهة المختصة وفق الموضوع، مع SLA لا يتجاوز 3 أيام عمل"
    return "الرد من قاعدة المعرفة وإغلاق الطلب مع تأكيد فهم المستفيد"


def _rule_enrich(category: str, body: str) -> Enrichment:
    sev, topic, reason = _rule_classify(category, body)
    return Enrichment(
        severity=sev,
        severity_reason=reason,
        topic_label=topic,
        recommended_action=_rule_action(category, body, sev, topic),
        source="rule",
    )


# helper to keep the fallback API used elsewhere unchanged
def _rule_severity(category: str, body: str) -> tuple[str, str]:
    sev, _, reason = _rule_classify(category, body)
    return sev, reason


# --- LLM-backed enrichment ---

_DOMAIN_CONTEXT = """السياق: مؤسسة تمويلية تُقدّم قروضاً وبرامج دعم لقطاع المستفيدين. الخدمات تشمل القروض، التمويل، برامج الدعم، السداد والتحصيل، خدمة المراجعين، والقنوات الرقمية.

أبرز إشارات الخطورة في طلبات المستفيدين: التعثّر في السداد، الاحتيال أو التلاعب، الخصم المالي الخاطئ، تأخر صرف المستحقات، رفض طلبات التمويل، طلبات الاسترداد العاجل."""

_ENRICH_SYSTEM = (
    "أنت محلل خدمة عملاء أول في مؤسسة تمويلية. تفهم منتجات المؤسسة "
    "وعملياتها. تستجيب فقط بصيغة JSON صالحة بالعربية، بدون أي شرح إضافي.\n\n"
    + _DOMAIN_CONTEXT
)

_ENRICH_PROMPT_TMPL = """صنّف هذا الطلب الوارد من المستفيد وأرجع JSON بالحقول التالية فقط:
- "severity": "عالية" أو "متوسطة" أو "منخفضة" — استخدم سياق المؤسسة التمويلية عند التقدير
- "severity_reason": تبرير محدّد يقتبس عبارة من نص الطلب ويوضّح الأثر التشغيلي/المالي على المستفيد. تجنّب التبريرات العامة.
- "topic_label": تسمية مختصرة (٢-٤ كلمات) للموضوع (مثل: «تعثّر في السداد»، «خصم خاطئ»، «استفسار عام»، «تحديث البيانات»)
- "recommended_action": إجراء واحد عملي قابل للتنفيذ، يذكر الجهة المختصة و SLA إن أمكن

الفئة المعلنة: {category}
نص الطلب: {body}

أعد JSON فقط بدون أي نص آخر."""


def enrich_record(category: str, body: str, *, prefer_llm: bool = True) -> Enrichment:
    """Classify a single record. Tries Groq → Ollama → rule-based."""
    if prefer_llm and (groq_available() or ollama_available()):
        try:
            raw, provider = _llm_generate(
                _ENRICH_PROMPT_TMPL.format(category=category or "", body=body or ""),
                system=_ENRICH_SYSTEM,
                temperature=0.1,
            )
            parsed = _try_parse_json(raw)
            if parsed:
                # Topic from rules used as fallback below
                _, rule_topic, rule_reason = _rule_classify(category, body) \
                    if hasattr(__import__('sys').modules[__name__], '_rule_classify') \
                    else (None, _rule_topic(category, body), _rule_severity(category, body)[1])
                sev = str(parsed.get("severity", "")).strip()
                if sev not in SEVERITY_VALUES:
                    sev, reason = _rule_severity(category, body)
                else:
                    reason = str(parsed.get("severity_reason") or "").strip() or rule_reason
                topic = str(parsed.get("topic_label") or "").strip() or rule_topic
                action = (
                    str(parsed.get("recommended_action") or "").strip()
                    or _rule_action(category, body, sev, topic)
                )
                return Enrichment(
                    severity=sev,
                    severity_reason=reason,
                    topic_label=topic,
                    recommended_action=action,
                    source=provider,
                )
        except Exception as exc:
            logger.warning("LLM enrich failed, falling back to rules: %s", exc)
    return _rule_enrich(category, body)


# --- Free-form Q&A over the dataset ---

_QA_SYSTEM = (
    "أنت مساعد تحليلي لخدمة العملاء. "
    "تفهم منتجات المؤسسة التمويلية وعملياتها كما هو مذكور أدناه. "
    "أجب باللغة العربية الرسمية، باختصار، بناءً على البيانات المُعطاة فقط. "
    "إن لم تكن المعلومة متوفرة، فاذكر ذلك صراحةً.\n\n"
    + _DOMAIN_CONTEXT
)


def answer_question(question: str, context_summary: str,
                    language: str = "ar") -> dict[str, Any]:
    """Answer a free-form analytical question against a pre-aggregated context.

    Tries Groq → Ollama → falls back to a structured echo of the summary.
    `language` ∈ {"ar", "en"} — controls the response language.
    """
    if groq_available() or ollama_available():
        try:
            instr = ("Answer concisely in English using professional language."
                     if language == "en" else "أجب باختصار وبصياغة مهنية.")
            label_q = "Question" if language == "en" else "السؤال"
            label_ctx = "Current data summary" if language == "en" else "ملخّص البيانات الحالية"
            raw, provider = _llm_generate(
                f"{label_q}: {question}\n\n{label_ctx}:\n{context_summary}\n\n{instr}",
                system=_QA_SYSTEM,
                temperature=0.2,
            )
            return {"answer": raw or "تعذّر توليد إجابة من النموذج.", "source": provider}
        except Exception as exc:
            logger.warning("LLM QA failed: %s", exc)

    msg_ar = ("لم يتم تفعيل أي نموذج لغوي. هذا ملخص البيانات المتاحة ضمن النطاق المحدد:\n\n"
              + context_summary)
    msg_en = ("No language model is configured. Here is a summary of the data in scope:\n\n"
              + context_summary)
    return {"answer": msg_en if language == "en" else msg_ar, "source": "rule"}


# ---------- corpus-level actionable insights ----------

_INSIGHTS_SYSTEM = (
    "أنت محلل بيانات أول في مؤسسة تمويلية. هدفك توليد رؤى عملية ملموسة "
    "قابلة للتنفيذ هذا الأسبوع، تستند إلى منتجات المؤسسة وعملياتها. "
    "كل رؤية تتضمن: ملاحظة موثّقة بالأرقام، إجراءً محدداً مع جهة مسؤولة، "
    "ومقياس متابعة قابل للقياس. تجنّب العموميات والكلام الإنشائي. "
    "أرجع JSON فقط بالعربية الرسمية.\n\n"
    + _DOMAIN_CONTEXT
)

_INSIGHTS_PROMPT = """فيما يلي ملخّص إحصائي مُكثّف لمشاركات المستفيدين خلال الفترة:

{signals}

استخرج {n} رؤى عملية بالضبط، كلٌّ منها قابلة للتنفيذ هذا الأسبوع. أعد JSON بالشكل:
{{
  "insights": [
    {{
      "title": "...",
      "evidence": "...",
      "action": "...",
      "metric": "..."
    }}
  ]
}}"""


def generate_insights(signals_text: str, n: int = 4) -> dict:
    """Actionable insights from a signals summary, via the provider chain."""
    if groq_available() or ollama_available():
        try:
            raw, provider = _llm_generate(
                _INSIGHTS_PROMPT.format(signals=signals_text, n=n),
                system=_INSIGHTS_SYSTEM,
                temperature=0.25,
            )
            parsed = _try_parse_json(raw)
            if parsed and isinstance(parsed.get("insights"), list):
                return {"insights": parsed["insights"][:n], "source": provider}
        except Exception as exc:
            logger.warning("LLM insights failed: %s", exc)
    return {"insights": [], "source": "rule"}
