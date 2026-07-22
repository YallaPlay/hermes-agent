from hermes_cli.config import DEFAULT_CONFIG


def test_continuation_preview_is_explicitly_disabled_by_default():
    assert DEFAULT_CONFIG["continuation_checkpoint"] == {
        "preview_enabled": False,
    }


def test_continuation_projector_has_an_independent_read_only_auxiliary_route():
    route = DEFAULT_CONFIG["auxiliary"]["continuation_checkpoint"]

    assert route == {
        "provider": "auto",
        "model": "",
        "base_url": "",
        "api_key": "",
        "timeout": 600,
        "extra_body": {},
        "reasoning_effort": "",
    }
