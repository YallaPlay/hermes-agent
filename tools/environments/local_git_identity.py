"""Attribute agent git commits to the authenticated session user.

When the session's ``HERMES_SESSION_USER_ID`` is an email (the ACP owner, e.g. a
Cloudflare Access login), child processes get ``GIT_AUTHOR_*`` for that user and
``GIT_COMMITTER_*`` from ``git.committer`` in config.yaml. Git env vars override
repo/global gitconfig without mutating it. Non-email ids (Slack, Telegram) and
sessions without an owner inject nothing, so their behavior is unchanged.
"""

import re

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f<>]")


def _clean(value: object) -> str:
    # Control chars or angle brackets would corrupt the commit header.
    return _CONTROL_RE.sub("", str(value or "")).strip()


def _name_from_email(email: str) -> str:
    local = email.split("@", 1)[0]
    return " ".join(p.capitalize() for p in re.split(r"[._\-+]+", local) if p) or local


def _read_committer_config() -> tuple[str, str]:
    try:
        from hermes_cli.config import load_config
        committer = ((load_config() or {}).get("git") or {}).get("committer") or {}
        return _clean(committer.get("name")), _clean(committer.get("email"))
    except Exception:
        return "", ""


def apply_git_identity_env(env: dict) -> None:
    """Inject author/committer identity into *env* in place. Explicit
    ``GIT_AUTHOR_*`` / ``GIT_COMMITTER_*`` values already present win."""
    email = _clean(env.get("HERMES_SESSION_USER_ID"))
    if not _EMAIL_RE.match(email):
        return
    name = _clean(env.get("HERMES_SESSION_USER_NAME")) or _name_from_email(email)
    env.setdefault("GIT_AUTHOR_NAME", name)
    env.setdefault("GIT_AUTHOR_EMAIL", email)
    committer_name, committer_email = _read_committer_config()
    if committer_name and committer_email:
        env.setdefault("GIT_COMMITTER_NAME", committer_name)
        env.setdefault("GIT_COMMITTER_EMAIL", committer_email)
