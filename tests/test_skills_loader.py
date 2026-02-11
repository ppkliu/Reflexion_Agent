"""Tests for SkillsLoader — markdown skill files with YAML frontmatter."""

import pytest
from pathlib import Path

from src.skills.loader import Skill, SkillsLoader


@pytest.fixture
def skills_dir(tmp_path):
    """Create a temporary skills directory with sample skill files."""
    # Skill 1: always_on
    (tmp_path / "reflexion.md").write_text(
        "---\n"
        "name: reflexion_guidelines\n"
        "description: Guidelines for reflection\n"
        "always_on: true\n"
        "tags: [reflexion, reasoning]\n"
        "---\n"
        "When reflecting, identify the failure point.\n"
        "Propose concrete fixes.\n",
        encoding="utf-8",
    )

    # Skill 2: not always_on
    (tmp_path / "tot_strategy.md").write_text(
        "---\n"
        "name: tot_search\n"
        "description: ToT search guidelines\n"
        "always_on: false\n"
        "tags: [tot, search]\n"
        "---\n"
        "Maximize diversity in branches.\n",
        encoding="utf-8",
    )

    # Skill 3: minimal frontmatter
    (tmp_path / "minimal.md").write_text(
        "---\n"
        "name: minimal_skill\n"
        "description: A minimal skill\n"
        "---\n"
        "Just some content.\n",
        encoding="utf-8",
    )

    # Invalid file: no frontmatter
    (tmp_path / "invalid.md").write_text(
        "This file has no YAML frontmatter.\n",
        encoding="utf-8",
    )

    return tmp_path


class TestSkillsLoader:

    def test_load_all(self, skills_dir):
        """load_all reads all valid .md files with frontmatter."""
        loader = SkillsLoader(skills_dir)
        skills = loader.load_all()

        # 3 valid skills (invalid.md has no frontmatter)
        assert len(skills) == 3
        assert len(loader) == 3

    def test_skill_content_parsed(self, skills_dir):
        """Skill content is extracted from after the frontmatter."""
        loader = SkillsLoader(skills_dir)
        loader.load_all()

        skill = loader.get_by_name("reflexion_guidelines")
        assert skill is not None
        assert skill.description == "Guidelines for reflection"
        assert skill.always_on is True
        assert "reflexion" in skill.tags
        assert "identify the failure point" in skill.content

    def test_always_on_filter(self, skills_dir):
        """get_always_on returns only always_on=True skills."""
        loader = SkillsLoader(skills_dir)
        loader.load_all()

        always_on = loader.get_always_on()
        assert len(always_on) == 1
        assert always_on[0].name == "reflexion_guidelines"

    def test_get_by_tags(self, skills_dir):
        """get_by_tags filters skills by tag membership."""
        loader = SkillsLoader(skills_dir)
        loader.load_all()

        tot_skills = loader.get_by_tags(["tot"])
        assert len(tot_skills) == 1
        assert tot_skills[0].name == "tot_search"

    def test_get_summaries(self, skills_dir):
        """get_summaries returns one-line-per-skill format."""
        loader = SkillsLoader(skills_dir)
        loader.load_all()

        summaries = loader.get_summaries()
        assert "reflexion_guidelines:" in summaries
        assert "tot_search:" in summaries
        assert "minimal_skill:" in summaries

    def test_build_always_on_prompt(self, skills_dir):
        """build_always_on_prompt concatenates always-on skill content."""
        loader = SkillsLoader(skills_dir)
        loader.load_all()

        prompt = loader.build_always_on_prompt()
        assert "## Skill: reflexion_guidelines" in prompt
        assert "identify the failure point" in prompt
        # Non-always-on skills should NOT be included
        assert "tot_search" not in prompt

    def test_to_prompt_block(self):
        """Skill.to_prompt_block formats correctly."""
        skill = Skill(
            name="test_skill",
            description="A test.",
            content="Do something useful.",
        )
        block = skill.to_prompt_block()
        assert "## Skill: test_skill" in block
        assert "A test." in block
        assert "Do something useful." in block

    def test_empty_directory(self, tmp_path):
        """Loading from empty directory returns empty list."""
        loader = SkillsLoader(tmp_path)
        skills = loader.load_all()
        assert skills == []
        assert loader.get_summaries() == "(no skills loaded)"

    def test_nonexistent_directory(self, tmp_path):
        """Loading from nonexistent directory returns empty list."""
        loader = SkillsLoader(tmp_path / "nonexistent")
        skills = loader.load_all()
        assert skills == []
