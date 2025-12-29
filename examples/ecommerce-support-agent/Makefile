clean-run:
	rm -rf ./.agent_chaos_runs
	uv sync --reinstall-package agent-chaos
	uv run agent-chaos run scenarios/
