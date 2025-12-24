from agent_chaos.scenario import CompletesWithin, MaxLLMCalls, Scenario, run_scenario


def test_driver_receives_ctx():
    got = {}

    def driver(ctx):
        got["has_ctx"] = ctx is not None
        return {"ok": True}

    scenario = Scenario(
        name="test-driver-ctx",
        agent=driver,
        assertions=[CompletesWithin(1.0), MaxLLMCalls(0)],
    )

    report = run_scenario(scenario, artifacts_dir=None, record_events=False)
    assert report.passed is True
    assert got.get("has_ctx") is True


def test_noarg_driver_supported():
    def driver():
        return {"ok": True}

    scenario = Scenario(
        name="test-driver-noarg",
        agent=driver,
        assertions=[CompletesWithin(1.0), MaxLLMCalls(0)],
    )

    report = run_scenario(scenario, artifacts_dir=None, record_events=False)
    assert report.passed is True


def test_expected_error_scenario_can_pass():
    from agent_chaos.scenario import ExpectError

    def driver(ctx):
        raise ValueError("boom")

    scenario = Scenario(
        name="test-expected-error",
        agent=driver,
        assertions=[ExpectError(r"ValueError")],
    )

    report = run_scenario(scenario, artifacts_dir=None, record_events=False)
    assert report.passed is True
    assert report.error is not None
