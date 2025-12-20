"""MCP Server for Simple Memory."""

import argparse
import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
)

from .config import get_config
from .memory import ConfigurationError, get_memory_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create MCP server
server = Server("simple-memory")


def check_configuration() -> tuple[bool, str]:
    """Check if the system is configured."""
    config = get_config()
    if not config.is_configured():
        return False, (
            "Simple Memory is not configured. Please configure the LLM and embedding "
            "settings via the web interface at http://localhost:8765/settings"
        )
    return True, ""


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="add_memory",
            description="Add a new memory to the knowledge base. The memory will be processed by LLM and stored with embeddings for semantic search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to remember",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "User identifier (optional, defaults to 'default')",
                        "default": "default",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional metadata to store with the memory",
                        "default": {},
                    },
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="search_memories",
            description="Search for relevant memories using semantic search. Returns memories similar to the query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5,
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Filter by user ID (optional)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags (optional)",
                    },
                    "min_similarity": {
                        "type": "number",
                        "description": "Minimum similarity percentage (0-100) to include in results. Uses config default if not specified.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_memory",
            description="Get a specific memory by its ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The memory ID",
                    },
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="list_memories",
            description="List all memories with optional filtering.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Filter by user ID (optional)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 20)",
                        "default": 20,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Offset for pagination (default: 0)",
                        "default": 0,
                    },
                },
            },
        ),
        Tool(
            name="update_memory",
            description="Update an existing memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The memory ID to update",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content (optional)",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "New metadata (optional)",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New tags (optional)",
                    },
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="delete_memory",
            description="Delete a memory by its ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The memory ID to delete",
                    },
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="get_memory_stats",
            description="Get statistics about the memory store.",
            inputSchema={
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "Filter by user ID (optional)",
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    # Check configuration first
    is_configured, error_msg = check_configuration()
    if not is_configured:
        return [TextContent(type="text", text=f"Error: {error_msg}")]

    try:
        memory_manager = get_memory_manager()

        if name == "add_memory":
            content = arguments["content"]
            user_id = arguments.get("user_id", "default")
            metadata = arguments.get("metadata", {})

            memory_id = await memory_manager.add_memory(
                content=content,
                user_id=user_id,
                metadata=metadata,
                process_with_llm=True,
            )

            return [
                TextContent(
                    type="text",
                    text=f"Memory added successfully with ID: {memory_id}",
                )
            ]

        elif name == "search_memories":
            query = arguments["query"]
            limit = arguments.get("limit", 5)
            user_id = arguments.get("user_id")
            tags = arguments.get("tags")
            min_similarity = arguments.get("min_similarity")

            results = await memory_manager.search(
                query=query,
                limit=limit,
                user_id=user_id,
                tags=tags,
                min_similarity=min_similarity,
            )

            if not results:
                return [TextContent(type="text", text="No memories found matching your query.")]

            # Format results
            formatted_results = []
            for i, mem in enumerate(results, 1):
                result = {
                    "id": mem.get("id"),
                    "content": mem.get("processed_content") or mem.get("content"),
                    "tags": mem.get("tags", []),
                    "created_at": mem.get("created_at"),
                    "similarity": mem.get("similarity"),
                    "rerank_score": mem.get("rerank_score"),
                }
                formatted_results.append(result)

            return [
                TextContent(
                    type="text",
                    text=json.dumps(formatted_results, indent=2, ensure_ascii=False),
                )
            ]

        elif name == "get_memory":
            memory_id = arguments["memory_id"]
            memory = await memory_manager.get_memory(memory_id)

            if not memory:
                return [TextContent(type="text", text=f"Memory not found: {memory_id}")]

            result = {
                "id": memory.get("id"),
                "content": memory.get("content"),
                "processed_content": memory.get("processed_content"),
                "tags": memory.get("tags", []),
                "metadata": memory.get("metadata", {}),
                "created_at": memory.get("created_at"),
                "updated_at": memory.get("updated_at"),
            }

            return [
                TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, ensure_ascii=False),
                )
            ]

        elif name == "list_memories":
            user_id = arguments.get("user_id")
            limit = arguments.get("limit", 20)
            offset = arguments.get("offset", 0)

            memories = await memory_manager.get_all_memories(
                user_id=user_id,
                limit=limit,
                offset=offset,
            )

            if not memories:
                return [TextContent(type="text", text="No memories found.")]

            formatted_results = []
            for mem in memories:
                result = {
                    "id": mem.get("id"),
                    "content": mem.get("processed_content") or mem.get("content"),
                    "tags": mem.get("tags", []),
                    "created_at": mem.get("created_at"),
                }
                formatted_results.append(result)

            return [
                TextContent(
                    type="text",
                    text=json.dumps(formatted_results, indent=2, ensure_ascii=False),
                )
            ]

        elif name == "update_memory":
            memory_id = arguments["memory_id"]
            content = arguments.get("content")
            metadata = arguments.get("metadata")
            tags = arguments.get("tags")

            success = await memory_manager.update_memory(
                memory_id=memory_id,
                content=content,
                metadata=metadata,
                tags=tags,
            )

            if success:
                return [TextContent(type="text", text=f"Memory {memory_id} updated successfully.")]
            else:
                return [TextContent(type="text", text=f"Failed to update memory: {memory_id}")]

        elif name == "delete_memory":
            memory_id = arguments["memory_id"]
            success = await memory_manager.delete_memory(memory_id)

            if success:
                return [TextContent(type="text", text=f"Memory {memory_id} deleted successfully.")]
            else:
                return [TextContent(type="text", text=f"Failed to delete memory: {memory_id}")]

        elif name == "get_memory_stats":
            user_id = arguments.get("user_id")

            count = await memory_manager.count_memories(user_id)
            tags = await memory_manager.get_all_tags()

            stats = {
                "total_memories": count,
                "unique_tags": len(tags),
                "tags": tags,
            }

            return [
                TextContent(
                    type="text",
                    text=json.dumps(stats, indent=2, ensure_ascii=False),
                )
            ]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except ConfigurationError as e:
        return [TextContent(type="text", text=f"Configuration Error: {str(e)}")]
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def run_stdio_server():
    """Run the MCP server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def run_sse_server(host: str = "0.0.0.0", port: int = 8766):
    """Run the MCP server with SSE transport for remote access."""
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import JSONResponse
    import uvicorn

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options(),
            )

    async def handle_messages(request):
        await sse.handle_post_message(request.scope, request.receive, request._send)

    async def health_check(request):
        config = get_config()
        return JSONResponse({
            "status": "ok",
            "configured": config.is_configured(),
            "server": "simple-memory",
        })

    app = Starlette(
        debug=False,
        routes=[
            Route("/health", health_check, methods=["GET"]),
            Route("/sse", handle_sse, methods=["GET"]),
            Route("/messages/", handle_messages, methods=["POST"]),
        ],
    )

    logger.info(f"Starting SSE MCP server at http://{host}:{port}")
    logger.info(f"SSE endpoint: http://{host}:{port}/sse")
    logger.info(f"Health check: http://{host}:{port}/health")

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Memory MCP Server")
    parser.add_argument(
        "--transport",
        "-t",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport type: stdio (default) or sse for remote access",
    )
    parser.add_argument(
        "--host",
        "-H",
        default="0.0.0.0",
        help="Host to bind SSE server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8766,
        help="Port for SSE server (default: 8766)",
    )

    args = parser.parse_args()

    if args.transport == "sse":
        asyncio.run(run_sse_server(args.host, args.port))
    else:
        asyncio.run(run_stdio_server())


if __name__ == "__main__":
    main()
