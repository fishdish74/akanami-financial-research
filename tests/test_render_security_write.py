from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from config import load_env_file
from obsidian import write_new_note_atomic
from renderer import deterministic_filename, render_article, validate_rendered_markdown
from schemas import ProcessingMetadata
from security import redact_secrets, safe_error
from tests.helpers import analysis, candidate


class RendererTests(unittest.TestCase):
    def test_render_is_deterministic_and_yaml_is_valid(self) -> None:
        item = candidate()
        payload = analysis()
        metadata = ProcessingMetadata(
            model_id="offline-smoke",
            prompt_version="market-analysis-v2",
            pipeline_version="v1",
        )
        first = render_article(payload, item, "NEW_EVENT", metadata)
        second = render_article(payload, item, "NEW_EVENT", metadata)
        self.assertEqual(first, second)
        frontmatter = validate_rendered_markdown(first)
        self.assertEqual(frontmatter["prompt_version"], "market-analysis-v2")
        self.assertTrue(frontmatter["needs_review"])

    def test_yaml_handles_unicode_colons_and_quotes(self) -> None:
        payload = analysis(title='NVIDIA: "Café" update')
        markdown = render_article(payload, candidate(title=payload.title), "NEW_EVENT")
        frontmatter = yaml.safe_load(markdown.split("---", 2)[1])
        self.assertEqual(frontmatter["source"], "Example Wire")
        self.assertIn('NVIDIA: "Café" update', markdown)

    def test_multiline_bullet_cannot_create_a_new_section(self) -> None:
        payload = analysis(reported_facts=["First line\n## injected heading"])
        markdown = render_article(payload, candidate(), "NEW_EVENT")
        self.assertIn("- First line\n  \\## injected heading", markdown)

    def test_active_obsidian_and_html_constructs_are_neutralized(self) -> None:
        hostile = (
            "[[Secret Note|Readable link]] ![[Embedded Note]] "
            "![pixel](obsidian://open?vault=Secret) `= dv.pages()` `$= app.vault` "
            "<% tp.file.title %> <script>alert(1)</script> <iframe\nsrc=x></iframe>\n"
            "---\n```dataview\nTABLE file.name\n```"
        )
        payload = analysis(
            title="[[Injected title]] <iframe src=x>",
            source="[[Injected source]]",
            reported_facts=[hostile],
        )
        markdown = render_article(payload, candidate(title=payload.title), "NEW_EVENT")
        body = markdown.split("---", 2)[2]
        frontmatter = yaml.safe_load(markdown.split("---", 2)[1])
        for active in ("[[", "![[", "`=", "`$=", "<%", "<script", "<iframe"):
            self.assertNotIn(active, markdown)
        self.assertNotRegex(body, r"(?m)^\s*```(?:dataview|dataviewjs)")
        self.assertNotRegex(body, r"(?m)^---$")
        self.assertIn("Readable link", body)
        self.assertIn("Embedded reference: Embedded Note", body)
        self.assertIn("Dataview expression: dv.pages()", body)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", body)
        self.assertEqual(frontmatter["source"], "Injected source")

    def test_filename_removes_unsafe_characters_and_is_bounded(self) -> None:
        payload = analysis(title=("Unsafe:/\\*?<>|\n" + "界" * 200))
        filename = deterministic_filename(payload)
        self.assertLessEqual(len(filename.encode("utf-8")), 220)
        for value in '<>:"/\\|?*\n':
            self.assertNotIn(value, filename)

    def test_validator_rejects_missing_section(self) -> None:
        markdown = render_article(analysis(), candidate(), "NEW_EVENT")
        with self.assertRaises(ValueError):
            validate_rendered_markdown(markdown.replace("## My Notes", "## Removed"))


class SecurityTests(unittest.TestCase):
    def test_configured_secret_is_redacted(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-secret-value"}):
            value = redact_secrets("request used sk-test-secret-value")
        self.assertNotIn("secret-value", value)
        self.assertIn("[REDACTED]", value)

    def test_key_shaped_text_is_redacted_without_config(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            value = redact_secrets("api_key=sk-example123456789")
        self.assertNotIn("example123456789", value)

    def test_safe_error_extracts_sanitized_sdk_metadata(self) -> None:
        class FakeError(Exception):
            status_code = 401
            body = {"error": {"type": "invalid_request_error", "code": "invalid_api_key", "message": "Bad sk-example123456789"}}

        value = safe_error(FakeError("fallback"), category="analysis")
        self.assertEqual(value["status"], 401)
        self.assertEqual(value["code"], "invalid_api_key")
        self.assertNotIn("example123456789", value["message"])

    def test_env_file_does_not_override_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text("OPENAI_MODEL=file-model\nNEW_SETTING='value'\n", encoding="utf-8")
            with patch.dict(os.environ, {"OPENAI_MODEL": "process-model"}, clear=True):
                names = load_env_file(path)
                self.assertEqual(os.environ["OPENAI_MODEL"], "process-model")
                self.assertEqual(os.environ["NEW_SETTING"], "value")
                self.assertEqual(names, {"OPENAI_MODEL", "NEW_SETTING"})


class WriteSafetyTests(unittest.TestCase):
    def test_disabled_write_fails_before_creating_directory(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "missing" / "note.md"
            with patch.dict(os.environ, {"MARKET_COLLECTOR_ENABLE_WRITES": "0"}):
                with self.assertRaises(PermissionError):
                    write_new_note_atomic(destination, "test", allowed_root=Path(folder))
            self.assertFalse(destination.parent.exists())

    def test_enabled_write_creates_new_note_in_temporary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "note.md"
            with patch.dict(os.environ, {"MARKET_COLLECTOR_ENABLE_WRITES": "1"}):
                write_new_note_atomic(destination, "hello\n", allowed_root=Path(folder))
            self.assertEqual(destination.read_text(encoding="utf-8"), "hello\n")

    def test_existing_file_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "note.md"
            destination.write_text("original", encoding="utf-8")
            with patch.dict(os.environ, {"MARKET_COLLECTOR_ENABLE_WRITES": "1"}):
                with self.assertRaises(FileExistsError):
                    write_new_note_atomic(destination, "replacement", allowed_root=Path(folder))
            self.assertEqual(destination.read_text(encoding="utf-8"), "original")

    def test_non_markdown_destination_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"MARKET_COLLECTOR_ENABLE_WRITES": "1"}):
                with self.assertRaises(ValueError):
                    write_new_note_atomic(
                        Path(folder) / "note.txt", "text", allowed_root=Path(folder)
                    )


if __name__ == "__main__":
    unittest.main()
