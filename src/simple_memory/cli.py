"""CLI entry point for Simple Memory."""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simple Memory - A memory MCP server using LanceDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  web       Start the web interface (default)
  mcp       Start the MCP server (stdio mode for local use)
  serve     Start the MCP server with SSE for remote access

Examples:
  simple-memory web              # Start web interface on default port
  simple-memory web -p 8080      # Start web interface on port 8080
  simple-memory mcp              # Start MCP server (stdio, for Claude Desktop)
  simple-memory serve            # Start SSE server for remote MCP access
  simple-memory serve -p 8766    # Start SSE server on custom port
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Web command
    web_parser = subparsers.add_parser("web", help="Start the web interface")
    web_parser.add_argument(
        "-p", "--port", type=int, default=8765, help="Port to run web server on (default: 8765)"
    )
    web_parser.add_argument(
        "-H", "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    web_parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # MCP command (stdio)
    subparsers.add_parser("mcp", help="Start the MCP server (stdio mode)")

    # Serve command (SSE for remote access)
    serve_parser = subparsers.add_parser("serve", help="Start MCP server with SSE for remote access")
    serve_parser.add_argument(
        "-p", "--port", type=int, default=8766, help="Port for SSE server (default: 8766)"
    )
    serve_parser.add_argument(
        "-H", "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )

    args = parser.parse_args()

    if args.command == "mcp":
        from .mcp_server import run_stdio_server
        import asyncio

        asyncio.run(run_stdio_server())

    elif args.command == "serve":
        from .mcp_server import run_sse_server
        import asyncio

        print(f"Starting Simple Memory MCP SSE server")
        print(f"SSE endpoint: http://{args.host}:{args.port}/sse")
        print(f"Health check: http://{args.host}:{args.port}/health")
        print("Press Ctrl+C to stop")
        asyncio.run(run_sse_server(args.host, args.port))

    elif args.command == "web":
        from .config import get_config_manager

        config_manager = get_config_manager()
        config = config_manager.load()

        # Update web config from CLI args
        config.web.port = args.port
        config.web.host = args.host
        config.web.debug = args.debug
        config_manager.save(config)

        from .web.app import app

        print(f"Starting Simple Memory web interface at http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        # Default to web if no command specified
        from .web.app import main as web_main

        print("Starting Simple Memory web interface at http://0.0.0.0:8765")
        print("Press Ctrl+C to stop")
        web_main()


if __name__ == "__main__":
    main()
