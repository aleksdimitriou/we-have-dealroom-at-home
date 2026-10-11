"""
Tests for src/llm.py (provider-neutral parts, no API calls) and src/golden.py.
Run from the project root:  python -m unittest discover tests
"""

import unittest

from src.golden import EXAMPLE_INPUT, clean_founder_name, first_url, read_input_rows
from src.llm import Usage, cost_usd, effort_options, inline_refs, parse_openrouter_response


class CostTests(unittest.TestCase):
    def test_sonnet_cost_counts_cache_writes_and_reads(self):
        usage = Usage(input_tokens=1_000_000, cache_write_tokens=1_000_000, cache_read_tokens=1_000_000,
                      output_tokens=1_000_000)
        self.assertAlmostEqual(cost_usd("claude-sonnet-5-5", usage), 2.00 + 2.50 + 0.20 + 10.00)

    def test_free_model_costs_nothing(self):
        self.assertEqual(cost_usd("qwen/qwen3.8-27b:free", Usage(input_tokens=10**6, output_tokens=10**6)), 0.0)

    def test_haiku_price(self):
        self.assertAlmostEqual(cost_usd("claude-haiku-4-5", Usage(input_tokens=10**6, output_tokens=10**6)), 6.00)

    def test_dated_model_ids_use_their_base_price(self):
        usage = Usage(input_tokens=10**6, output_tokens=10**6)
        self.assertAlmostEqual(cost_usd("claude-haiku-4-5-20251001", usage), cost_usd("claude-haiku-4-5", usage))
        self.assertGreater(cost_usd("claude-unknown-20251001", usage), 10.0)       # still conservative when unknown

    def test_effort_is_left_out_for_haiku_only(self):
        self.assertEqual(effort_options("claude-haiku-4-5", "medium"), {})
        self.assertEqual(effort_options("claude-sonnet-5-5", "medium"), {"output_config": {"effort": "medium"}})

    def test_unknown_model_is_priced_conservatively(self):
        self.assertGreater(cost_usd("some-new-model", Usage(output_tokens=1_000_000)), 10.0)


class SchemaTests(unittest.TestCase):
    def test_inline_refs(self):
        schema = {"type": "object", "properties": {"x": {"$ref": "#/$defs/X"}},
                  "$defs": {"X": {"type": "string", "enum": ["a"]}}}
        self.assertEqual(inline_refs(schema), {"type": "object", "properties": {"x": {"type": "string", "enum": ["a"]}}})


class OpenRouterParseTests(unittest.TestCase):
    def test_tool_call_and_usage(self):
        data = {"model": "qwen/qwen3.8-27b", "choices": [{"finish_reason": "tool_calls", "message": {
            "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "fetch_page", "arguments": '{"url": "https://a.com", "for_subject": "company"}'}}]}}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 80, "prompt_tokens_details": {"cached_tokens": 200}}}
        turn = parse_openrouter_response(data, "qwen/qwen3.8-27b:free")
        self.assertEqual(turn.tool_calls[0].input, {"url": "https://a.com", "for_subject": "company"})
        self.assertEqual((turn.usage.input_tokens, turn.usage.cache_read_tokens, turn.usage.output_tokens), (1000, 200, 80))
        self.assertEqual(turn.cost_usd, 0.0)

    def test_bad_arguments_become_none(self):
        data = {"choices": [{"message": {"tool_calls": [{"id": "c1", "function": {
            "name": "submit_claims", "arguments": "{not json"}}]}}]}
        self.assertIsNone(parse_openrouter_response(data, "m:free").tool_calls[0].input)


class GoldenTests(unittest.TestCase):
    def test_clean_founder_name(self):
        self.assertEqual(clean_founder_name("Jonas (Joni) Weber"), "Jonas Weber")
        self.assertEqual(clean_founder_name("Sam Okafor (registered as Samuel Okafor)"), "Sam Okafor")

    def test_first_url(self):
        self.assertEqual(first_url("https://www.antler.co/portfolio (filters: Sector = Industrials)"),
                         "https://www.antler.co/portfolio")
        self.assertIsNone(first_url("none"))

    def test_a_row_from_the_example_input(self):
        rows = {row.company_name: (row, raw) for row, raw in read_input_rows(EXAMPLE_INPUT)}
        row, raw = rows["Example Robotics"]
        self.assertEqual(row.company_url, "https://www.example-robotics.com")
        self.assertEqual(row.accelerator_profile_url, "https://www.ycombinator.com/companies/example-robotics")
        self.assertEqual(row.founder_names, ["Jane Doe", "John Roe"])
        self.assertEqual(raw[1], "John Roe (Johnny)")
        foods, foods_raw = rows["Example Foods"]
        self.assertEqual(foods.accelerator_profile_url, "https://www.antler.co/portfolio")          # the note is cut
        self.assertEqual((foods.founder_names, foods_raw), (["Jane Doe"], ["Jane Doe (registered as Janet Doe)"]))
