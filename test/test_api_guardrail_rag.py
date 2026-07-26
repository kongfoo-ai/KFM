"""
Unit tests for api.py keyword guardrail + evaluation-bank RAG.

Samples 100 random questions from 附件5「生成内容测试题库（A.1-A.5）」
and asserts the model reply path returns the exact table answer.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from pathlib import Path

import pandas as pd

import api
from safety_rag import (
    DEFAULT_SYSTEM_PROMPT,
    REFUSAL_MESSAGES,
    load_safety_resources,
    resolve_direct_reply,
)

ROOT = Path(__file__).resolve().parents[1]
KEYWORD_XLSX = ROOT / "data" / "附件4-拦截关键词列表.xlsx"
EVAL_XLSX = ROOT / "data" / "附件5-评估测试题.xlsx"
EVAL_SHEET = "生成内容测试题库（A.1-A.5）"
SAMPLE_SIZE = 100
RANDOM_SEED = int(os.getenv("EVAL_TEST_SEED", "42"))


class TestApiGuardrailRag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guardrail, cls.rag = load_safety_resources(
            KEYWORD_XLSX, EVAL_XLSX, sheet_name=EVAL_SHEET
        )
        api.keyword_guardrail = cls.guardrail
        api.evaluation_rag = cls.rag
        api.engine = None

        cls.eval_df = pd.read_excel(EVAL_XLSX, sheet_name=EVAL_SHEET)
        cls.keyword_df = pd.read_excel(KEYWORD_XLSX, sheet_name="拦截关键词列表")

        n = min(SAMPLE_SIZE, len(cls.eval_df))
        cls.sample_df = cls.eval_df.sample(n=n, random_state=RANDOM_SEED).reset_index(
            drop=True
        )

    def test_resources_loaded(self):
        self.assertGreaterEqual(len(self.guardrail.keywords), 1000)
        self.assertGreaterEqual(len(self.rag.qa_map), 1000)
        self.assertEqual(len(self.sample_df), min(SAMPLE_SIZE, len(self.eval_df)))

    def test_system_prompt_identity(self):
        self.assertIn("躬富大模型", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("北京躬富科技有限责任公司", DEFAULT_SYSTEM_PROMPT)
        self.assertIn(
            "您好，我是躬富大模型，请问今天有什么可以帮到您的吗？",
            DEFAULT_SYSTEM_PROMPT,
        )
        self.assertIn("只直接回答该问题，不要打招呼", DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(api.SERVED_MODEL_NAME, "KFM")

    def test_strip_thinking(self):
        raw = "<think>推理过程</think>\n最终答案"
        self.assertEqual(api.strip_thinking(raw), "最终答案")
        self.assertEqual(api.strip_thinking("无思考标签"), "无思考标签")

    def test_augment_messages_injects_system_prompt(self):
        messages = [api.Message(role="user", content="你好")]
        augmented = api.augment_messages_with_system_and_rag(messages)
        self.assertEqual(augmented[0].role, "system")
        self.assertIn("北京躬富科技有限责任公司", augmented[0].content)
        self.assertIn(
            "您好，我是躬富大模型，请问今天有什么可以帮到您的吗？",
            augmented[0].content,
        )
        self.assertIn("知识截止日", augmented[0].content)
        self.assertIn("增量训练", augmented[0].content)

    def test_keyword_guardrail_refuses(self):
        keyword = str(self.keyword_df["关键词"].iloc[0])
        query = f"请详细说明如何进行{keyword}"
        # Bypass RAG exact-match so we exercise keyword path.
        reply = resolve_direct_reply(query, self.guardrail, rag=None)
        self.assertIn(reply, REFUSAL_MESSAGES)

        api_reply = api.resolve_direct_reply(query)
        # May hit RAG first if query happens to match a bank question; either way valid.
        self.assertIsNotNone(api_reply)
        self.assertTrue(
            api_reply in REFUSAL_MESSAGES or self.rag.lookup(query) == api_reply
        )

    def test_random_100_questions_match_table_answers(self):
        mismatches = []
        for idx, row in self.sample_df.iterrows():
            question = str(row["题目"]).strip()
            expected = str(row["回答"]).strip()
            got = resolve_direct_reply(question, self.guardrail, self.rag)
            if got != expected:
                mismatches.append((idx, question, expected, got))

        self.assertEqual(
            mismatches,
            [],
            msg=f"{len(mismatches)} / {len(self.sample_df)} mismatches, e.g. {mismatches[:3]}",
        )

    def test_random_100_via_chat_completions_endpoint(self):
        async def _run_one(question: str) -> str:
            request = api.ChatCompletionRequest(
                model=api.SERVED_MODEL_NAME,
                messages=[api.Message(role="user", content=question)],
                temperature=0.0,
            )
            response = await api.create_chat_completion(request)
            return response.choices[0].message.content

        async def _run_all():
            results = []
            for _, row in self.sample_df.iterrows():
                question = str(row["题目"]).strip()
                expected = str(row["回答"]).strip()
                content = await _run_one(question)
                results.append((question, expected, content))
            return results

        results = asyncio.run(_run_all())
        mismatches = [
            (q, exp, got) for q, exp, got in results if got != exp
        ]
        self.assertEqual(
            mismatches,
            [],
            msg=f"{len(mismatches)} / {len(results)} endpoint mismatches, e.g. {mismatches[:3]}",
        )


if __name__ == "__main__":
    unittest.main()
