from agent_chaos.scenario import (
    BaselineScenario,
    CompletesWithin,
    MaxLLMCalls,
    Turn,
    run_scenario,
)


def test_driver_receives_ctx():
    got = {}

    def driver(ctx, turn_input):
        got["has_ctx"] = ctx is not None
        return {"ok": True}

    scenario = BaselineScenario(
        name="test-driver-ctx",
        description="Test that driver receives context",
        agent=driver,
        turns=[Turn("test")],
        assertions=[CompletesWithin(1.0), MaxLLMCalls(0)],
    )

    report = run_scenario(scenario, artifacts_dir=None, record_events=False)
    assert report.passed is True
    assert got.get("has_ctx") is True


def test_flexible_signature_driver_supported():
    """Drivers can use *args to accept ctx and turn_input."""

    def driver(*args):
        return {"ok": True}

    scenario = BaselineScenario(
        name="test-driver-flexible",
        description="Test that flexible-signature driver is supported",
        agent=driver,
        turns=[Turn("test")],
        assertions=[CompletesWithin(1.0), MaxLLMCalls(0)],
    )

    report = run_scenario(scenario, artifacts_dir=None, record_events=False)
    assert report.passed is True


def test_expected_error_scenario_can_pass():
    from agent_chaos.scenario import ExpectError

    def driver(ctx, turn_input):
        raise ValueError("boom")

    scenario = BaselineScenario(
        name="test-expected-error",
        description="Test that expected errors can pass",
        agent=driver,
        turns=[Turn("test")],
        assertions=[ExpectError(r"ValueError")],
    )

    report = run_scenario(scenario, artifacts_dir=None, record_events=False)
    assert report.passed is True
    assert report.error is not None
