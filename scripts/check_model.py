from pathlib import Path

from kdb_tools import config
from pydantic import BaseModel
from pydantic_ai import Agent


class ProjectInfo(BaseModel):
    name: str
    purpose: str


def main():
    cfg = config.load(Path("kdb.toml"))

    agent = Agent(
        config.model_from(cfg),
        output_type=ProjectInfo,
        model_settings=config.settings_from(cfg),
    )

    @agent.tool_plain
    def get_project_name() -> str:
        """Return the registered project name."""
        print("[TOOL EXECUTED] get_project_name")
        return "Cortex KDB"

    result = agent.run_sync(
    "Use the get_project_name tool to obtain the project name. "
    "Its purpose is to compile research papers into "
    "an evidence-grounded knowledge database. "
    "Return the project name and purpose."
)

    print("Model:", cfg.model)
    print("Context:", cfg.context)
    print("Response:")
    print(result.output.model_dump_json(indent=2))
    print("Token usage:")
    print(result.usage)


if __name__ == "__main__":
    main()