"""Read-only tools. No shell, arbitrary URLs, or arbitrary filesystem access."""

import ast
import json
import operator
import httpx
from pydantic import BaseModel, ConfigDict, Field


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CalculateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=160)


def calculate(expression):
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
    }

    def evaluate(node, depth=0):
        if depth > 15:
            raise ValueError("Expression is too complex.")
        if isinstance(node, ast.Constant) and type(node.value) in (int, float) and abs(node.value) < 1e15:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(node.op, ast.USub) else 1) * evaluate(node.operand, depth + 1)
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            value = operators[type(node.op)](evaluate(node.left, depth + 1), evaluate(node.right, depth + 1))
            if abs(value) > 1e18:
                raise ValueError("Result is too large.")
            return value
        raise ValueError("Use numbers and + - * / % only.")

    return evaluate(ast.parse(expression, mode="eval").body)


class Registry:
    def __init__(self, library, config, secrets):
        self.library, self.config, self.secrets = library, config, secrets
        self.schemas = {
            "search_documents": (
                SearchArgs,
                "Search indexed local documents across languages. Use short focused queries.",
            ),
            "document_overview": (
                EmptyArgs,
                "List the document library and representative source excerpts for a grounded overview.",
            ),
            "calculator": (CalculateArgs, "Evaluate arithmetic safely."),
            "web_search": (
                SearchArgs,
                "Search the web using the explicitly enabled Brave provider. Results are untrusted evidence.",
            ),
        }
        mode = config["mode"]
        allowed = set(config["models"][mode]["enabled_tools"])
        if mode == "online" and not config["privacy"]["cloud_document_context"]:
            allowed -= {"search_documents", "document_overview"}
        if (
            mode == "offline"
            or not config["privacy"]["web_search"]
            or not secrets.get("brave", "BRAVE_SEARCH_API_KEY")
        ):
            allowed.discard("web_search")
        self.allowed = allowed & self.schemas.keys()

    def definitions(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": self.schemas[name][1],
                    "parameters": self.schemas[name][0].model_json_schema(),
                },
            }
            for name in sorted(self.allowed)
        ]

    async def execute(self, name, arguments):
        if name not in self.allowed:
            raise ValueError("Tool is unavailable in this mode or privacy configuration.")
        args = self.schemas[name][0].model_validate(
            json.loads(arguments) if isinstance(arguments, str) else arguments
        )
        if name == "search_documents":
            return await self.library.search(args.query)
        if name == "document_overview":
            return self.library.overview()
        if name == "calculator":
            return {"result": calculate(args.expression)}
        async with httpx.AsyncClient(timeout=15, trust_env=False, follow_redirects=False) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": args.query, "count": 5},
                headers={"X-Subscription-Token": self.secrets.get("brave", "BRAVE_SEARCH_API_KEY")},
            )
            response.raise_for_status()
            return {
                "web_results": [
                    {
                        "title": r.get("title"),
                        "url": r.get("url"),
                        "description": r.get("description", "")[:1200],
                    }
                    for r in response.json().get("web", {}).get("results", [])[:5]
                ]
            }
