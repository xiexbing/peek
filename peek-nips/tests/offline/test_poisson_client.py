# Copyright 2026 Anonymous Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for benchmarks/poisson_client.py

Verifies:
1. The poisson client can load workloads independently (no server needed).
2. Request generation is deterministic -- two runs with the same params
   produce byte-identical request sequences.
"""

import pytest

pytestmark = pytest.mark.gpu
import unittest

from peek.offline.benchmarks.poisson_client import load_requests, flush_cache


class TestLoadRequestsDeterministic(unittest.TestCase):
    """load_requests must produce identical output across two calls."""

    def test_shared_system_prompts_deterministic(self):
        r1 = load_requests("shared_system_prompts", n=20, num_groups=3,
                           system_prompt_len=64)
        r2 = load_requests("shared_system_prompts", n=20, num_groups=3,
                           system_prompt_len=64)
        self.assertEqual(len(r1), len(r2))
        for a, b in zip(r1, r2):
            self.assertEqual(a["id"], b["id"])
            self.assertEqual(a["token_ids"], b["token_ids"])

    def test_few_shot_mmlu_deterministic(self):
        r1 = load_requests("few_shot_mmlu", n=20)
        r2 = load_requests("few_shot_mmlu", n=20)
        self.assertEqual(len(r1), len(r2))
        for a, b in zip(r1, r2):
            self.assertEqual(a["id"], b["id"])
            self.assertEqual(a["token_ids"], b["token_ids"])

    def test_returns_nonempty(self):
        reqs = load_requests("shared_system_prompts", n=10, num_groups=2,
                             system_prompt_len=32)
        self.assertEqual(len(reqs), 10)
        for r in reqs:
            self.assertIn("id", r)
            self.assertIn("token_ids", r)
            self.assertGreater(len(r["token_ids"]), 0)

    def test_request_ids_unique(self):
        reqs = load_requests("shared_system_prompts", n=50, num_groups=5,
                             system_prompt_len=64)
        ids = [r["id"] for r in reqs]
        self.assertEqual(len(ids), len(set(ids)))


class TestFlushCache(unittest.TestCase):
    def test_flush_returns_false_on_connection_error(self):
        self.assertFalse(flush_cache("http://localhost:99999"))


if __name__ == "__main__":
    unittest.main()
