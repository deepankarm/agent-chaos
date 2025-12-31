clean-reinstall:
	rm -rf ./.agent_chaos_runs
	uv sync --reinstall-package agent-chaos

clean-run: clean-reinstall
	uv run agent-chaos run scenarios/fuzzing.py
