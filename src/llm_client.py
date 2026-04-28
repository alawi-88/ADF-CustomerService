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
    severity_reason_ar: str
    severity_reason_en: str
    topic_label_ar: str
    topic_label_en: str
    recommended_action_ar: str
    recommended_action_en: str
    source: str             # "llm" | "rule"

    # Compat shims so older callers that read .topic_label / .severity_reason /
    # .recommended_action still get the Arabic version (the system's primary).
    @property
    def topic_label(self) -> str: return self.topic_label_ar
    @property
    def severity_reason(self) -> str: return self.severity_reason_ar
    @property
    def recommended_action(self) -> str: return self.recommended_action_ar


SEVERITY_VALUES = ("عالية", "متوسطة", "منخفضة")
SEVERITY_EN = {"عالية": "High", "متوسطة": "Medium", "منخفضة": "Low"}
CATEGORY_EN = {
    "شكوى": "Complaint",
    "استفسار": "Inquiry",
    "اقتراح": "Suggestion",
    "دعم فني": "Tech support",
    "خدمة مراجع": "Beneficiary service",
}


# --- Rule-based engine ---
# The patterns below are tuned to financial-services customer support:
# loans/financing, repayment & collections, beneficiary subsidies, and
# digital channels. They produce severity, topic, and recommended action
# for each request without requiring any LLM.

# Patterns: phrase fragments → (severity, topic_hint, business interpretation)
# Order matters: more specific patterns first.
# Each pattern: (keywords, topic_ar, topic_en, reason_ar_template, reason_en_template)
_HIGH_PATTERNS = [
    (("متعسر", "تعسر", "تعثر"),
     "تعثّر في السداد", "Default in repayment",
     "إشارة إلى حالة تعثّر في السداد، وهي من أعلى المخاطر التشغيلية وتستوجب تواصلاً مع فريق التحصيل.",
     "Indicates a default in repayment — one of the highest operational risks; requires escalation to the collections team."),
    (("احتيال", "تزوير", "تلاعب"),
     "بلاغ احتيال", "Fraud report",
     "إشارة محتملة إلى احتيال أو تلاعب، تتطلب تصعيداً فورياً لإدارة المخاطر والامتثال.",
     "Possible fraud or tampering — requires immediate escalation to risk and compliance."),
    (("خصم خاطئ", "خصم زائد", "حُسم", "حسم خاطئ"),
     "خصم مالي مُتنازَع عليه", "Disputed deduction",
     "ادعاء بخصم مالي غير صحيح من حساب المستفيد، يستلزم مراجعة فورية لفك الإشكال.",
     "Claim of an incorrect deduction from the beneficiary's account — needs immediate review."),
    (("ايقاف الحسم", "إيقاف الحسم", "إيقاف الخصم", "ايقاف الخصم"),
     "إيقاف الخصم على القرض", "Loan deduction halt",
     "طلب إيقاف الخصم على قرض قائم — قرار مالي حساس يؤثر على جدولة السداد.",
     "Request to halt deductions on an active loan — a sensitive financial decision affecting repayment schedule."),
    (("متأخر", "تأخر", "تأخير", "لم يُصرف", "لم يصرف", "بدون رد", "بلا رد"),
     "تأخر في الإجراء", "Delay in processing",
     "ادعاء بتأخر في معالجة طلب أو صرف مالي، يؤثر مباشرةً على المستفيد ويتطلب متابعة عاجلة.",
     "Claim of a processing or disbursement delay — directly affects the beneficiary and needs urgent follow-up."),
    (("رفض القرض", "رفض الطلب", "تم رفض"),
     "رفض طلب", "Application rejected",
     "اعتراض على رفض طلب تمويل، يتطلب توضيحاً لأسباب القرار وفق السياسة.",
     "Objection to a rejected financing request — requires clarification of the policy-based decision."),
    (("استرداد", "استرجاع المبلغ"),
     "طلب استرداد", "Refund request",
     "طلب استرداد مبلغ مالي، يتطلب التحقق من سجل المعاملة وإجراء التسوية.",
     "Refund request — requires verifying the transaction record and reconciling."),
    (("عاجل", "مستعجل", "ضروري"),
     "طلب عاجل", "Urgent request",
     "وُصف الطلب بأنه عاجل من قِبل المستفيد، يستحق التعامل ضمن مسار الأولوية.",
     "Beneficiary marked the request as urgent — should be handled in the priority queue."),
]

_MEDIUM_PATTERNS = [
    (("سداد", "تسديد", "قسط", "أقساط"),
     "السداد والأقساط", "Repayment & instalments",
     "استعلام أو طلب يخصّ جدولة السداد أو الأقساط، يستحق متابعة لتفادي التحوّل إلى تعثّر.",
     "Inquiry or request about the repayment schedule or instalments — worth following up to prevent default."),
    (("قرض", "تمويل"),
     "القروض والتمويل", "Loans & financing",
     "طلب يخصّ منتج تمويلي، يتطلب توضيحاً من فريق الائتمان.",
     "Question about a financing product — requires clarification from the credit team."),
    (("دعم زراعي", "دعم المزارعين", "دعم الإنتاج"),
     "الدعم الزراعي", "Agricultural support",
     "طلب يخصّ برامج الدعم، يستحق إحالة إلى الجهة المختصة.",
     "Question about support programs — should be routed to the responsible unit."),
    (("دواجن", "ماشية", "أبقار", "أغنام", "نحل", "أسماك"),
     "الإنتاج الحيواني", "Livestock production",
     "طلب يخصّ نشاط إنتاج حيواني ضمن نطاق الصندوق.",
     "Question about a livestock production activity in the program scope."),
    (("مشروع", "مزرعة"),
     "مشاريع المستفيدين", "Beneficiary projects",
     "طلب يخصّ مشروعاً قائماً للمستفيد، يستحق متابعة للحفاظ على استمراريته.",
     "Question about an active beneficiary project — worth following up to preserve continuity."),
    (("تحديث بيانات", "تحديث رقم", "تحديث الجوال"),
     "تحديث البيانات", "Data update",
     "طلب تحديث بيانات تواصل أو شخصية، إجراء روتيني لكن يلزم لإكمال الخدمات.",
     "Contact / personal data update — routine, but required to complete services."),
    (("بوابة", "تطبيق", "موقع", "نظام"),
     "القنوات الرقمية", "Digital channels",
     "طلب يخصّ تجربة الاستخدام في القنوات الرقمية.",
     "Question about user experience on the digital channels."),
]

_LOW_PATTERNS = [
    (("شكر", "تقدير", "ممتاز"),
     "إشادة وثناء", "Praise / thanks",
     "إشادة من المستفيد، لا تستوجب إجراءً تصحيحياً.",
     "Praise from the beneficiary — no corrective action required."),
    (("اقتراح", "مقترح", "تطوير"),
     "اقتراح تطوير", "Improvement suggestion",
     "اقتراح تحسين من المستفيد، يُحوَّل إلى سجل التحسينات.",
     "Improvement suggestion — to be logged in the improvements register."),
    (("استعلام", "استفسار", "كيف", "متى", "ما هي"),
     "استعلام عام", "General inquiry",
     "طلب معلومة عامة، يمكن الرد عليه من قاعدة المعرفة دون تصعيد.",
     "General information request — can be answered from the knowledge base without escalation."),
]


def _excerpt(body: str, limit: int = 60) -> str:
    """Short, clean excerpt of the body for use in justifications."""
    s = (body or "").strip()
    if not s:
        return ""
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit].rstrip() + "…"


def _match_pattern(text: str, patterns: list) -> tuple | None:
    for keys, topic_ar, topic_en, reason_ar, reason_en in patterns:
        for k in keys:
            if k in text:
                return (k, topic_ar, topic_en, reason_ar, reason_en)
    return None


def _rule_classify(category: str, body: str) -> dict:
    """Return a dict with severity + topic+reason in both AR and EN.

    Body excerpts are kept verbatim (the "quotes" exception in EN mode):
    we do not translate beneficiary content.
    """
    cat = (category or "").strip()
    cat_en = CATEGORY_EN.get(cat, cat)
    body_clean = (body or "").strip()
    text = f"{cat} {body_clean}"
    excerpt = _excerpt(body_clean)

    def _join(reason: str, lang: str) -> str:
        if not excerpt:
            return reason
        suffix = f" نص الطلب: «{excerpt}»." if lang == "ar" else f" Request text: «{excerpt}»."
        return reason + suffix

    # 1) High-severity signals
    hit = _match_pattern(text, _HIGH_PATTERNS)
    if hit:
        _, t_ar, t_en, r_ar, r_en = hit
        return {
            "severity": "عالية",
            "topic_ar": t_ar, "topic_en": t_en,
            "reason_ar": _join(r_ar, "ar"), "reason_en": _join(r_en, "en"),
        }

    # 2) Category-driven default for complaints
    if cat == "شكوى":
        if excerpt:
            return {
                "severity": "عالية",
                "topic_ar": "الشكاوى العامة", "topic_en": "General complaints",
                "reason_ar": f"شكوى مباشرة من المستفيد بشأن «{excerpt}» — تتطلب تصعيداً وتواصلاً استباقياً للحفاظ على رضا المتعامل.",
                "reason_en": f"Direct complaint from the beneficiary about «{excerpt}» — requires escalation and proactive outreach to preserve customer satisfaction.",
            }
        return {
            "severity": "عالية",
            "topic_ar": "الشكاوى العامة", "topic_en": "General complaints",
            "reason_ar": "شكوى مباشرة بدون وصف تفصيلي — تستدعي اتصالاً استباقياً لاستيضاح المشكلة.",
            "reason_en": "Direct complaint without details — needs a proactive call to clarify the issue.",
        }

    # 3) Medium signals
    hit = _match_pattern(text, _MEDIUM_PATTERNS)
    if hit:
        _, t_ar, t_en, r_ar, r_en = hit
        return {
            "severity": "متوسطة",
            "topic_ar": t_ar, "topic_en": t_en,
            "reason_ar": _join(r_ar, "ar"), "reason_en": _join(r_en, "en"),
        }

    if cat == "دعم فني":
        return {
            "severity": "متوسطة",
            "topic_ar": "الدعم الفني للقنوات الرقمية", "topic_en": "Digital-channel tech support",
            "reason_ar": f"طلب دعم فني{(' بشأن «' + excerpt + '»') if excerpt else ''} — يؤثر على قدرة المستفيد على إكمال خدمته.",
            "reason_en": f"Tech-support request{(' regarding «' + excerpt + '»') if excerpt else ''} — affects the beneficiary's ability to complete their service.",
        }

    if cat == "خدمة مراجع":
        return {
            "severity": "متوسطة",
            "topic_ar": "خدمة المراجعين", "topic_en": "Beneficiary service",
            "reason_ar": f"طلب يخصّ مسار خدمة المراجعين{(' حول «' + excerpt + '»') if excerpt else ''} — يتطلب متابعة لاستكمال إجراءات المستفيد.",
            "reason_en": f"Beneficiary-service request{(' about «' + excerpt + '»') if excerpt else ''} — needs follow-up to complete the beneficiary's procedures.",
        }

    # 4) Low signals
    hit = _match_pattern(text, _LOW_PATTERNS)
    if hit:
        _, t_ar, t_en, r_ar, r_en = hit
        return {
            "severity": "منخفضة",
            "topic_ar": t_ar, "topic_en": t_en,
            "reason_ar": _join(r_ar, "ar"), "reason_en": _join(r_en, "en"),
        }

    if cat in ("استفسار", "اقتراح"):
        if cat == "استفسار":
            t_ar, t_en = "استعلام عام", "General inquiry"
        else:
            t_ar, t_en = "اقتراح تطوير", "Improvement suggestion"
        r_ar = (f"طلب من نوع «{cat}»{(' حول «' + excerpt + '»') if excerpt else ''} — "
                "لا يستوجب تصعيداً ويمكن الرد ضمن المسار الاعتيادي.")
        r_en = (f"A «{cat_en}» type request{(' about «' + excerpt + '»') if excerpt else ''} — "
                "does not require escalation and can be handled in the regular flow.")
        return {
            "severity": "منخفضة",
            "topic_ar": t_ar, "topic_en": t_en,
            "reason_ar": r_ar, "reason_en": r_en,
        }

    # 5) Default
    return {
        "severity": "متوسطة",
        "topic_ar": cat or "غير مصنّف",
        "topic_en": cat_en or "Uncategorised",
        "reason_ar": f"تصنيف افتراضي{(' للطلب بشأن «' + excerpt + '»') if excerpt else ''} — لم تُكتشف إشارات صريحة تحدد مستوى الأولوية.",
        "reason_en": f"Default classification{(' for the request about «' + excerpt + '»') if excerpt else ''} — no explicit signals detected to determine priority.",
    }


def _rule_topic(category: str, body: str) -> str:
    return _rule_classify(category, body)["topic_ar"]


_ACTION_BY_CATEGORY_AND_SEVERITY = {
    ("شكوى", "عالية"):    ("تصعيد فوري إلى مختص الشكاوى — اتصال خلال 24 ساعة + تسجيل في نظام الجودة",
                            "Immediate escalation to the complaints specialist — call within 24 hours + log in the QA system"),
    ("شكوى", "متوسطة"):   ("إحالة إلى فريق رعاية المستفيدين مع SLA 48 ساعة",
                            "Route to the beneficiary care team with a 48-hour SLA"),
    ("دعم فني", "عالية"):  ("تذكرة دعم فوري + التحقق من توفّر الخدمة الرقمية",
                            "Immediate support ticket + verify availability of the digital service"),
    ("دعم فني", "متوسطة"): ("فتح تذكرة دعم فني والرد بحلّ موثّق",
                            "Open a tech-support ticket and reply with a documented fix"),
    ("استفسار", "منخفضة"): ("الرد من قاعدة المعرفة وإغلاق الطلب مع تأكيد الفهم",
                            "Reply from the knowledge base and close the request, confirming understanding"),
    ("استفسار", "متوسطة"): ("تحويل إلى المختص للتأكد من المعلومة قبل الرد",
                            "Route to the specialist to verify the information before replying"),
    ("اقتراح", "منخفضة"):  ("تسجيل المقترح في سجل التحسينات وإحالته إلى لجنة التطوير",
                            "Log the suggestion in the improvements register and refer it to the development committee"),
    ("خدمة مراجع", "متوسطة"): ("تحويل إلى مسار خدمة المراجعين وإبلاغ المستفيد بالمتطلبات الناقصة",
                                "Route to the reviewer-service track and notify the beneficiary of any missing requirements"),
    ("خدمة مراجع", "عالية"):  ("خدمة مراجعين عاجلة — متابعة المستفيد ومراجعة الملف",
                                "Urgent reviewer service — follow up with the beneficiary and review the file"),
}


def _rule_action(category: str, body: str, severity: str, topic: str) -> tuple[str, str]:
    cat = (category or "").strip()
    base = _ACTION_BY_CATEGORY_AND_SEVERITY.get((cat, severity))
    if base:
        return base
    if severity == "عالية":
        if "تعثّر" in topic or "تعثر" in topic:
            return ("إحالة فورية إلى فريق التحصيل وتفعيل خطة استرداد مرنة وفق وضع المستفيد",
                    "Immediate referral to the collections team and activation of a flexible recovery plan tailored to the beneficiary")
        if "احتيال" in topic:
            return ("تصعيد إلى إدارة المخاطر والامتثال خلال ساعة + إيقاف العمليات المرتبطة",
                    "Escalate to risk and compliance within one hour + suspend the related operations")
        if "خصم" in topic:
            return ("مراجعة سجل الخصم خلال 24 ساعة وإجراء التسوية المالية إن ثبت الخطأ",
                    "Review the deduction record within 24 hours and reconcile the amount if an error is confirmed")
        if "تأخر" in topic:
            return ("متابعة الطلب لدى الجهة المختصة واتصال استباقي بالمستفيد بنتيجة المراجعة",
                    "Follow up with the responsible unit and proactively contact the beneficiary with the outcome")
    if severity == "متوسطة":
        return ("إحالة إلى الجهة المختصة وفق الموضوع، مع SLA لا يتجاوز 3 أيام عمل",
                "Route to the responsible unit by topic, with an SLA of no more than 3 business days")
    return ("الرد من قاعدة المعرفة وإغلاق الطلب مع تأكيد فهم المستفيد",
            "Reply from the knowledge base and close the request, confirming the beneficiary's understanding")


def _rule_enrich(category: str, body: str) -> Enrichment:
    cls = _rule_classify(category, body)
    act_ar, act_en = _rule_action(category, body, cls["severity"], cls["topic_ar"])
    return Enrichment(
        severity=cls["severity"],
        severity_reason_ar=cls["reason_ar"],
        severity_reason_en=cls["reason_en"],
        topic_label_ar=cls["topic_ar"],
        topic_label_en=cls["topic_en"],
        recommended_action_ar=act_ar,
        recommended_action_en=act_en,
        source="rule",
    )


def _rule_severity(category: str, body: str) -> tuple[str, str]:
    cls = _rule_classify(category, body)
    return cls["severity"], cls["reason_ar"]


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
    """Classify a single record. Tries Groq → Ollama → rule-based.

    Always returns AR + EN versions of every text field. If the LLM only
    supplies one language, the other is filled from the rule engine.
    """
    if prefer_llm and (groq_available() or ollama_available()):
        try:
            raw, provider = _llm_generate(
                _ENRICH_PROMPT_TMPL.format(category=category or "", body=body or ""),
                system=_ENRICH_SYSTEM,
                temperature=0.1,
            )
            parsed = _try_parse_json(raw)
            if parsed:
                rule = _rule_enrich(category, body)
                sev = str(parsed.get("severity", "")).strip()
                if sev not in SEVERITY_VALUES:
                    sev = rule.severity
                topic_ar = (str(parsed.get("topic_label") or "").strip() or rule.topic_label_ar)
                reason_ar = (str(parsed.get("severity_reason") or "").strip() or rule.severity_reason_ar)
                action_ar = (str(parsed.get("recommended_action") or "").strip() or rule.recommended_action_ar)
                return Enrichment(
                    severity=sev,
                    severity_reason_ar=reason_ar,
                    severity_reason_en=rule.severity_reason_en,
                    topic_label_ar=topic_ar,
                    topic_label_en=rule.topic_label_en,
                    recommended_action_ar=action_ar,
                    recommended_action_en=rule.recommended_action_en,
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


# ---------- group-level intent synthesis ----------

_GROUP_SYSTEM = (
    "أنت موظف خدمة عملاء أول في مؤسسة تمويلية. تفهم منتجات المؤسسة كما هو "
    "مذكور أدناه. عندك عدة طلبات من مستفيدين تشترك في موضوع واحد. مهمتك أن "
    "تستنبط ما يطلبه المستفيدون فعلاً (وليس النص الحرفي) وأن تحدّد أفضل "
    "إجراء عملي لموظف خدمة العملاء. ردّ بصيغة JSON صالحة فقط.\n\n"
    + _DOMAIN_CONTEXT
)

_GROUP_PROMPT_TMPL = """فيما يلي عيّنة من {n} طلب وردت إلى مركز خدمة العملاء، وكلها تشترك في موضوع واحد:

{samples}

سياق إضافي:
- الفئة الغالبة: {top_category}
- نسبة الخطورة العالية في الموضوع: {high_pct}%
- لغة الإجابة المطلوبة: {language_label}

اشرح بدقة ما يطلبه المستفيدون فعلاً (وليس الحرفي)، ثم أعطِ إجراءً واحداً عملياً قابلاً للتنفيذ من قِبل موظف خدمة العملاء.

أعد JSON فقط بالحقول التالية:
{{
  "intent": "ما يطلبه المستفيد فعلاً، جملة واحدة كاملة في سياق المؤسسة التمويلية.",
  "employee_response": "إجراء واحد ومحدّد للموظف. اذكر الجهة المختصة وSLA إن أمكن.",
  "rationale": "سبب موجز يربط بين النصوص أعلاه وما استنبطته."
}}"""


def enrich_group(samples: list[str], top_category: str, high_pct: float,
                 language: str = "ar") -> dict | None:
    """Ask the LLM to interpret a cluster of related complaints in domain
    context. Returns None if no LLM is reachable or parsing failed."""
    if not (groq_available() or ollama_available()):
        return None
    if not samples:
        return None
    sample_block = "\n".join(f"- {s}" for s in samples[:8])
    lang_label = "العربية" if language == "ar" else "English"
    try:
        raw, provider = _llm_generate(
            _GROUP_PROMPT_TMPL.format(
                n=len(samples), samples=sample_block,
                top_category=top_category or "—",
                high_pct=int(high_pct),
                language_label=lang_label,
            ),
            system=_GROUP_SYSTEM,
            temperature=0.2,
        )
        parsed = _try_parse_json(raw)
        if not parsed:
            return None
        return {
            "intent":            str(parsed.get("intent") or "").strip(),
            "employee_response": str(parsed.get("employee_response") or "").strip(),
            "rationale":         str(parsed.get("rationale") or "").strip(),
            "provider":          provider,
        }
    except Exception as exc:
        logger.warning("group enrich failed: %s", exc)
        return None


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
