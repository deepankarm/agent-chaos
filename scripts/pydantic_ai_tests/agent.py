from __future__ import annotations

from agent_chaos import ChaosContext
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

# EXTRA_HEADERS = {
#     "x-aig-user-id": "pydantic-ai-scenario",
#     "x-aig-team-name": "agent-chaos",
# }
# MODEL_NAME = os.getenv("ANTHROPIC_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0")


# def get_anthropic_model() -> AnthropicModel:
#     import anthropic

#     return AnthropicModel(
#         model_name=MODEL_NAME,
#         provider=AnthropicProvider(
#             anthropic_client=anthropic.AsyncAnthropic(default_headers=EXTRA_HEADERS)
#         ),
#     )


def get_anthropic_model() -> AnthropicModel:
    return AnthropicModel(model_name="claude-sonnet-4-5-20250929")


class WeatherDeps:
    def __init__(self):
        self.location_cache: dict = {}


async def get_weather(ctx, city: str) -> str:
    """Get weather for a city."""
    weather_data = {
        "tokyo": {"temp": 22, "condition": "sunny"},
        "london": {"temp": 12, "condition": "rainy"},
        "new york": {"temp": 18, "condition": "cloudy"},
        "sydney": {"temp": 28, "condition": "sunny"},
        "paris": {"temp": 15, "condition": "partly cloudy"},
        "berlin": {"temp": 10, "condition": "overcast"},
    }
    w = weather_data.get(city.lower())
    if not w:
        return f"Weather in {city}: 20°C, partly cloudy"
    return f"Weather in {city}: {w['temp']}°C, {w['condition']}"


async def suggest_activity(ctx, weather_description: str) -> str:
    """Suggest activity based on weather."""
    wl = weather_description.lower()
    if "rainy" in wl:
        return "Indoor activity recommended: visit a museum or cafe."
    if "sunny" in wl:
        return "Outdoor activity recommended: hiking, cycling, or picnic."
    if "cloudy" in wl or "overcast" in wl:
        return "Light outdoor activity: casual stroll or outdoor cafe."
    return "Plan based on conditions."


def create_weather_agent() -> Agent[WeatherDeps, str]:
    agent = Agent[WeatherDeps, str](
        model=get_anthropic_model(),
        deps_type=WeatherDeps,
        system_prompt=(
            "You are a helpful weather assistant.\n"
            "Always call get_weather first, then call suggest_activity.\n"
            "Keep answers concise."
        ),
    )
    agent.tool(get_weather)
    agent.tool(suggest_activity)
    return agent


async def run_agent(ctx: ChaosContext, query: str):
    """Single parameterized agent runner for all scenarios."""
    agent = create_weather_agent()
    deps = WeatherDeps()
    result = await agent.run(query, deps=deps)
    # Extract just the text content from the response
    if hasattr(result, "output"):
        return str(result.output)
    # Fallback for different pydantic-ai versions
    resp = result.response if hasattr(result, "response") else result
    if hasattr(resp, "parts") and resp.parts:
        return resp.parts[0].content
    return str(resp)


async def run_agent_streaming(ctx: ChaosContext, query: str):
    """Agent runner with streaming for stream chaos tests."""
    agent = create_weather_agent()
    deps = WeatherDeps()
    from pydantic_ai import FinalResultEvent

    output = ""
    async with agent.iter(query, deps=deps) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        if isinstance(event, FinalResultEvent):
                            async for text in request_stream.stream_text():
                                # output += text
                                print(f"[Text] {text}")

            elif Agent.is_end_node(node):
                print(f"[Done] {run.result.output}")
                output += run.result.output

    return str(output)
