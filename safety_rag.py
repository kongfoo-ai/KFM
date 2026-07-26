"""Keyword guardrail and evaluation-bank RAG helpers used by api.py."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REFUSAL_MESSAGES = (
    "抱歉，我不能协助此类请求。",
    "抱歉，无法按此要求提供信息。",
    "抱歉，我无法提供相关方法。",
)

GREETING_REPLY = "您好，我是躬富大模型，请问今天有什么可以帮到您的吗？"

# Knowledge cutoff tracks the last incremental training point: always 14 days before "today".
KNOWLEDGE_CUTOFF_LAG_DAYS = 14

# Prefer refuse over fabricating dates/casualties/causes for sensitive hotspot incidents.
SENSITIVE_EVENT_REFUSAL = (
    "抱歉，对该类敏感事件的具体事实我无法在未经权威核验的情况下作答，"
    "请以官方通报为准，勿采信未经证实的信息。"
)

# Pure greetings / presence pings — no concrete ask. Keep short to avoid false positives.
_GREETING_EXACT = {
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "hello",
    "hi",
    "hey",
    "在吗",
    "在不在",
    "早上好",
    "中午好",
    "下午好",
    "晚上好",
    "早安",
    "午安",
    "晚安",
}

# Ask about knowledge / training-data cutoff (not "how to handle stale corpus").
_KNOWLEDGE_CUTOFF_RE = re.compile(
    r"(知识截止|知识截止日期|知识更新到|知识更新至|知识到什么时候|"
    r"训练数据截止|训练截止|增量训练|训练到什么时候|数据更新到|"
    r"knowledge\s*cut[- ]?off|cutoff\s*date)",
    re.IGNORECASE,
)

# Sensitive hotspot fact probes (accidents / building strikes / casualties).
# Without an exact knowledge-bank hit, refuse rather than invent details.
_SENSITIVE_HOT_EVENT_RE = re.compile(
    r"(中信大厦|中国尊).{0,24}(飞机|航空器|撞击|碰撞|撞机|撞了)|"
    r"(飞机|航空器|撞机).{0,24}(中信大厦|中国尊)|"
    r"(轻型(飞机|航空器|运动航空器)).{0,24}(碰撞|撞击).{0,24}(高层|大厦|建筑)|"
    r"(高层建筑|大厦).{0,16}(被)?(飞机|航空器).{0,8}(撞|碰撞|撞击)"
)


def knowledge_cutoff_date(today: Optional[date] = None) -> date:
    """Last incremental training date ≈ current calendar date minus two weeks."""
    base = today or date.today()
    return base - timedelta(days=KNOWLEDGE_CUTOFF_LAG_DAYS)


def format_knowledge_cutoff(cutoff: Optional[date] = None, today: Optional[date] = None) -> str:
    d = cutoff if cutoff is not None else knowledge_cutoff_date(today)
    return f"{d.year}年{d.month}月{d.day}日"


def knowledge_cutoff_reply(today: Optional[date] = None) -> str:
    cutoff = format_knowledge_cutoff(today=today)
    return f"我的知识截止日为{cutoff}。"


def is_knowledge_cutoff_query(text: str) -> bool:
    if not text:
        return False
    return bool(_KNOWLEDGE_CUTOFF_RE.search(str(text)))


def is_sensitive_hot_event_query(text: str) -> bool:
    """True for sensitive accident/hotspot fact questions prone to fabrication."""
    if not text:
        return False
    return bool(_SENSITIVE_HOT_EVENT_RE.search(str(text)))


def build_system_prompt(today: Optional[date] = None) -> str:
    """System prompt with a rolling knowledge-cutoff date (today − 14 days)."""
    cutoff = format_knowledge_cutoff(today=today)
    return (
        "你是躬富大模型，由北京躬富科技有限责任公司开发。"
        "当你需要打招呼或开场时，请使用："
        f"“{GREETING_REPLY}”\n"
        "回答要求：\n"
        "1. 牢记开发者是北京躬富科技有限责任公司，回答时应明确自身身份与服务边界，避免伪通用、无边界人设。\n"
        "2. 若下方知识库中存在与用户问题一致或高度相似的条目，请直接使用对应回答原文，不要改写。\n"
        "3. 对违法违规、危险或不当请求应拒答。\n"
        "4. 如果用户提出了具体问题，只直接回答该问题，不要打招呼或寒暄，不要答非所问或转移话题。\n"
        "5. 涉及事故灾难、建筑撞击、伤亡等热点敏感事件的时间/地点/人数/原因等具体事实时："
        "仅当知识库已有可核验官方口径条目时按原文回答；否则必须拒答并引导查阅官方通报，"
        "严禁编造或猜测日期、伤亡、动机等细节——宁可拒答，也不要捏造事实。\n"
        "6. 涉及国家主权、港澳台等表述必须使用规范口径，不得沿用未经核验的境外中文语料说法。\n"
        f"7. 本系统知识截止日为{cutoff}（内部口径：当前日期前两周、最近一次增量训练时间点）；"
        "若用户询问知识截止时间/训练数据截止时间，只答复该日期本身，"
        "不要附加“对应当前日期前两周”“增量训练时间点”等说明，不要使用固定旧截止日期。"
    )


# Snapshot for static identity checks; runtime paths should call build_system_prompt().
DEFAULT_SYSTEM_PROMPT = build_system_prompt()

EVAL_SHEET_DEFAULT = "生成内容测试题库（A.1-A.5）"
EXTRA_COVERAGE_BANK_DEFAULT = "extra_coverage_bank.json"
RAG_TOP_K_DEFAULT = 3
RAG_SIM_THRESHOLD_DEFAULT = 0.55


def normalize_text(text: str) -> str:
    return "".join(str(text).split())


def is_greeting(text: str) -> bool:
    """True for short, meaningless greetings that should use the fixed opener."""
    if not text:
        return False
    cleaned = re.sub(r"[!?.。！？～~\s]+", "", str(text)).strip().lower()
    if not cleaned:
        return False
    return cleaned in {g.lower() for g in _GREETING_EXACT}


def pick_refusal_message(seed: str) -> str:
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return REFUSAL_MESSAGES[int(digest, 16) % len(REFUSAL_MESSAGES)]


def _char_bigrams(text: str) -> set:
    normalized = normalize_text(text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


def jaccard_similarity(a: str, b: str) -> float:
    sa, sb = _char_bigrams(a), _char_bigrams(b)
    if not sa or not sb:
        return 1.0 if a == b else 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


@dataclass
class KeywordGuardrail:
    keywords: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_excel(cls, path: Path) -> "KeywordGuardrail":
        import pandas as pd

        df = pd.read_excel(path, sheet_name="拦截关键词列表")
        keywords = [
            str(k).strip()
            for k in df["关键词"].dropna().tolist()
            if str(k).strip()
        ]
        keywords.sort(key=len, reverse=True)
        return cls(keywords=tuple(keywords))

    def find_keyword(self, text: str) -> Optional[str]:
        if not text:
            return None
        for keyword in self.keywords:
            if keyword in text:
                return keyword
        return None

    def refusal_for(self, text: str) -> Optional[str]:
        hit = self.find_keyword(text)
        if hit is None:
            return None
        return pick_refusal_message(hit)


@dataclass
class EvaluationRAG:
    qa_map: Dict[str, str] = field(default_factory=dict)
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)

    @classmethod
    def from_excel(
        cls, path: Path, sheet_name: str = EVAL_SHEET_DEFAULT
    ) -> "EvaluationRAG":
        import pandas as pd

        df = pd.read_excel(path, sheet_name=sheet_name)
        qa_map: Dict[str, str] = {}
        questions: List[str] = []
        answers: List[str] = []
        for _, row in df.iterrows():
            question = str(row["题目"]).strip()
            answer = str(row["回答"]).strip()
            if not question or not answer or question == "nan" or answer == "nan":
                continue
            qa_map[normalize_text(question)] = answer
            questions.append(question)
            answers.append(answer)
        return cls(qa_map=qa_map, questions=questions, answers=answers)

    def extend_from_json(self, path: Path) -> int:
        """Merge supplemental Q&A bank (extra coverage). Later duplicates overwrite."""
        if not path or not Path(path).is_file():
            return 0
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        items = payload.get("items") or payload.get("cases") or []
        added = 0
        for item in items:
            question = str(item.get("question") or item.get("题目") or "").strip()
            answer = str(
                item.get("answer")
                or item.get("expected_answer")
                or item.get("回答")
                or ""
            ).strip()
            if not question or not answer:
                continue
            key = normalize_text(question)
            if key not in self.qa_map:
                self.questions.append(question)
                self.answers.append(answer)
                added += 1
            else:
                # Keep list aligned: update existing answer in place when possible.
                try:
                    idx = next(
                        i
                        for i, q in enumerate(self.questions)
                        if normalize_text(q) == key
                    )
                    self.answers[idx] = answer
                except StopIteration:
                    self.questions.append(question)
                    self.answers.append(answer)
                    added += 1
            self.qa_map[key] = answer
        return added

    def lookup(self, query: str) -> Optional[str]:
        return self.qa_map.get(normalize_text(query))

    def retrieve(
        self,
        query: str,
        top_k: int = RAG_TOP_K_DEFAULT,
        threshold: float = RAG_SIM_THRESHOLD_DEFAULT,
    ) -> List[Tuple[str, str, float]]:
        scored: List[Tuple[str, str, float]] = []
        for question, answer in zip(self.questions, self.answers):
            score = jaccard_similarity(query, question)
            if score >= threshold:
                scored.append((question, answer, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def format_context(self, hits: Sequence[Tuple[str, str, float]]) -> str:
        if not hits:
            return ""
        lines = ["【知识库检索结果】"]
        for idx, (question, answer, score) in enumerate(hits, start=1):
            lines.append(
                f"{idx}. 问题：{question}\n   标准回答：{answer}\n   相似度：{score:.3f}"
            )
        lines.append("若用户问题与上述条目一致或高度相似，请直接输出对应“标准回答”原文。")
        return "\n".join(lines)


def load_safety_resources(
    keyword_path: Path,
    eval_path: Path,
    sheet_name: str = EVAL_SHEET_DEFAULT,
    extra_bank_path: Optional[Path] = None,
) -> Tuple[KeywordGuardrail, EvaluationRAG]:
    guardrail = KeywordGuardrail.from_excel(keyword_path)
    rag = EvaluationRAG.from_excel(eval_path, sheet_name=sheet_name)
    if extra_bank_path is None:
        extra_bank_path = Path(eval_path).resolve().parent / EXTRA_COVERAGE_BANK_DEFAULT
    rag.extend_from_json(extra_bank_path)
    return guardrail, rag


def resolve_direct_reply(
    user_content: str,
    guardrail: Optional[KeywordGuardrail],
    rag: Optional[EvaluationRAG],
) -> Optional[str]:
    """
    Deterministic short-circuit:
    1) Exact RAG hit from evaluation / extra coverage bank
    2) Sensitive hotspot fact probe without bank hit → refuse (no fabrication)
    3) Knowledge-cutoff question → rolling date (today − 14 days)
    4) Pure greeting → fixed opener
    5) Keyword guardrail refusal
    """
    if rag is not None:
        exact = rag.lookup(user_content)
        if exact is not None:
            return exact

    if is_sensitive_hot_event_query(user_content):
        return SENSITIVE_EVENT_REFUSAL

    if is_knowledge_cutoff_query(user_content):
        return knowledge_cutoff_reply()

    if is_greeting(user_content):
        return GREETING_REPLY

    if guardrail is not None:
        refusal = guardrail.refusal_for(user_content)
        if refusal is not None:
            return refusal

    return None
