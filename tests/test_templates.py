"""Tests for prompt templates."""

from self_correct import templates


class TestBuiltins:
    def test_every_builtin_has_a_description_and_body(self):
        for name, (description, body) in templates.BUILTIN_TEMPLATES.items():
            assert description.strip(), f"{name} has no description"
            assert body.strip(), f"{name} has no body"

    def test_get_returns_a_builtin(self):
        assert templates.get_template("factual-summary")

    def test_unknown_template_is_none(self):
        assert templates.get_template("no-such-template") is None


class TestPlaceholders:
    def test_finds_names_in_order_without_duplicates(self):
        found = templates.placeholders("Compare $a and $b, again $a, then ${c}.")
        assert found == ["a", "b", "c"]

    def test_no_placeholders(self):
        assert templates.placeholders("A prompt with no variables.") == []

    def test_every_builtin_placeholder_is_renderable(self):
        for name, (_, body) in templates.BUILTIN_TEMPLATES.items():
            values = {key: "X" for key in templates.placeholders(body)}
            rendered, missing = templates.render(body, values)
            assert missing == [], f"{name} reported missing keys after filling all"
            assert "$" not in rendered, f"{name} left a placeholder unfilled"


class TestRender:
    def test_fills_values(self):
        rendered, missing = templates.render("Summarise $topic.", {"topic": "bees"})
        assert rendered == "Summarise bees."
        assert missing == []

    def test_refuses_when_a_value_is_missing(self):
        """A literal '$topic' must never reach the model."""
        rendered, missing = templates.render("Compare $a and $b.", {"a": "x"})
        assert rendered is None
        assert missing == ["b"]


class TestUserTemplates:
    def test_user_template_is_listed(self, tmp_path, monkeypatch):
        monkeypatch.setenv(templates.TEMPLATE_DIR_ENV, str(tmp_path))
        (tmp_path / "my-check.txt").write_text("Check $thing.", encoding="utf-8")

        names = [row[0] for row in templates.list_templates()]
        assert "my-check" in names
        assert templates.get_template("my-check") == "Check $thing."

    def test_user_template_overrides_a_builtin(self, tmp_path, monkeypatch):
        monkeypatch.setenv(templates.TEMPLATE_DIR_ENV, str(tmp_path))
        (tmp_path / "timeline.txt").write_text("Mine for $topic.", encoding="utf-8")

        assert templates.get_template("timeline") == "Mine for $topic."
        sources = {row[0]: row[1] for row in templates.list_templates()}
        assert "overrides" in sources["timeline"]

    def test_missing_directory_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv(templates.TEMPLATE_DIR_ENV, str(tmp_path / "absent"))
        assert templates.user_templates() == {}
        assert templates.list_templates(), "built-ins still listed"
