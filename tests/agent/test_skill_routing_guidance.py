"""Rendered skill instructions respect scoped project exceptions, not operational shortcuts."""

import pytest

from agent import prompt_builder as pb
from agent.oneshot_footprint import ONESHOT_SKILLS_LOAD_GUIDANCE


@pytest.fixture
def routing_profile(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    skills = home / "skills"
    for name, description in (
        ("safe-ops", "Use before operational work."),
        ("topic-reference", "Background about the same topic."),
    ):
        path = skills / "general" / name / "SKILL.md"
        path.parent.mkdir(parents=True)
        path.write_text(
            f"---\nname: {name}\ndescription: {description}\n---\nProcedure.\n",
            encoding="utf-8",
        )
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.chdir(project)
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    yield home, project
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)


def test_project_fast_path_coexists_with_required_skill_recipe(routing_profile):
    home, project = routing_profile
    rules = (
        "# Routing\n"
        "For a simple question answer directly if one read-only lookup suffices; no skill load is required.\n"
        "This exception excludes operational workflows, writes, and production access.\n"
        "For operational work, load safe-ops before action.\n"
    )
    (project / "AGENTS.md").write_text(rules, encoding="utf-8")
    context = pb.build_context_files_prompt(cwd=str(project), skip_soul=True, home_override=home)
    skills = pb.build_skills_system_prompt(skills_dir_override=home / "skills")

    assert rules.strip() in context
    assert "Follow explicit project skill recipes and load their required skills" in skills
    assert "Honor an explicit project simple-question fast path only within its stated scope" in skills
    assert "never use it to bypass operational or safety prerequisites" in skills
    assert "Related-topic skills are suggestions, not blanket requirements" in skills
    assert "even partially relevant" not in skills
    assert "Only proceed without loading a skill if genuinely none are relevant" not in skills
    assert "- safe-ops: Use before operational work." in skills
    assert "- topic-reference: Background about the same topic." in skills


def test_no_project_exception_keeps_applicable_loads_and_oneshot_variant(routing_profile, monkeypatch):
    home, project = routing_profile
    assert pb.build_context_files_prompt(cwd=str(project), skip_soul=True, home_override=home) == ""
    skills = pb.build_skills_system_prompt(available_tools={"skill_view", "terminal"},
                                          skills_dir_override=home / "skills")

    assert "Load skills that apply to the actual procedure" in skills
    assert "including mandatory operational and safety guidance" in skills
    assert "Without such an exception, load applicable skills before proceeding" in skills
    assert "even when terminal seems sufficient" in skills
    assert "skill_view(name)" in skills

    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    oneshot = pb.build_skills_system_prompt(available_tools={"skill_view", "terminal"},
                                           skills_dir_override=home / "skills")
    assert oneshot.split("<available_skills>", 1)[0] == ONESHOT_SKILLS_LOAD_GUIDANCE + "\n"
    assert oneshot.split("<available_skills>", 1)[1].split("</available_skills>", 1)[0] == (
        skills.split("<available_skills>", 1)[1].split("</available_skills>", 1)[0]
    )
