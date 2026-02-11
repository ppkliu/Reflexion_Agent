"""SkillsLoader — load markdown skill files with YAML frontmatter.

Mirrors nanobot's SkillsLoader pattern:
  - Skills are defined as .md files in a skills directory.
  - Each file has YAML frontmatter (name, description, always_on, tags).
  - The body contains the skill content (instructions/prompts).
  - always_on skills are injected into every agent loop iteration.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Regex to split YAML frontmatter from markdown body
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z",
    re.DOTALL,
)


@dataclass
class Skill:
    """A loaded skill definition."""

    name: str
    description: str
    always_on: bool = False
    content: str = ""
    tags: list[str] = field(default_factory=list)
    source_path: Path | None = None

    def to_prompt_block(self) -> str:
        """Format this skill as a prompt injection block."""
        return f"## Skill: {self.name}\n{self.description}\n\n{self.content}"


class SkillsLoader:
    """Load and manage skill files from a directory.

    Usage:
        loader = SkillsLoader(Path("skills/"))
        loader.load_all()

        # Get skills that should always be injected
        always_on = loader.get_always_on()

        # Get a specific skill
        skill = loader.get_by_name("reflexion_guidelines")

        # Build prompt block from all always-on skills
        prompt = loader.build_always_on_prompt()
    """

    def __init__(self, skills_dir: Path | str | None = None) -> None:
        if skills_dir is None:
            # Default: project_root/skills/
            skills_dir = Path(__file__).parent.parent.parent / "skills"
        self._dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}

    @property
    def skills_dir(self) -> Path:
        return self._dir

    def load_all(self) -> list[Skill]:
        """Load all .md skill files from the skills directory.

        Returns:
            List of loaded Skill objects.
        """
        self._skills.clear()

        if not self._dir.exists():
            logger.warning("Skills directory does not exist: %s", self._dir)
            return []

        md_files = sorted(self._dir.glob("*.md"))
        loaded: list[Skill] = []

        for path in md_files:
            skill = self._parse_skill_file(path)
            if skill:
                self._skills[skill.name] = skill
                loaded.append(skill)
                logger.debug("Loaded skill: %s from %s", skill.name, path.name)

        logger.info("Loaded %d skills from %s", len(loaded), self._dir)
        return loaded

    def get_by_name(self, name: str) -> Skill | None:
        """Get a skill by its name."""
        return self._skills.get(name)

    def get_by_tags(self, tags: list[str]) -> list[Skill]:
        """Get skills matching any of the given tags."""
        tag_set = set(tags)
        return [s for s in self._skills.values() if tag_set & set(s.tags)]

    def get_always_on(self) -> list[Skill]:
        """Return skills marked as always_on=True."""
        return [s for s in self._skills.values() if s.always_on]

    def get_summaries(self) -> str:
        """Return a compact summary of all loaded skills (one line each).

        Format: "- skill_name: description"
        """
        lines = []
        for skill in self._skills.values():
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines) if lines else "(no skills loaded)"

    def build_always_on_prompt(self) -> str:
        """Build a combined prompt block from all always-on skills."""
        skills = self.get_always_on()
        if not skills:
            return ""
        blocks = [s.to_prompt_block() for s in skills]
        return "\n\n".join(blocks)

    def list_names(self) -> list[str]:
        """Return names of all loaded skills."""
        return list(self._skills.keys())

    def _parse_skill_file(self, path: Path) -> Skill | None:
        """Parse a single .md skill file with YAML frontmatter."""
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.error("Failed to read skill file %s: %s", path, e)
            return None

        match = _FRONTMATTER_RE.match(raw)
        if not match:
            logger.warning(
                "Skill file %s has no valid YAML frontmatter (---...---), skipping.",
                path.name,
            )
            return None

        frontmatter_str, body = match.group(1), match.group(2)

        try:
            meta: dict[str, Any] = yaml.safe_load(frontmatter_str) or {}
        except yaml.YAMLError as e:
            logger.error("Invalid YAML in %s: %s", path.name, e)
            return None

        name = meta.get("name", path.stem)
        description = meta.get("description", "")
        always_on = bool(meta.get("always_on", False))
        tags = meta.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        return Skill(
            name=name,
            description=description,
            always_on=always_on,
            content=body.strip(),
            tags=tags,
            source_path=path,
        )

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills
