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
  mcp       Start the MCP server

Examples:
  simple-memory web          # Start web interface on default port
  simple-memory web -p 8080  # Start web interface on port 8080
  simple-memory mcp          # Start MCP server (for Claude Desktop)
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

    # MCP command
    subparsers.add_parser("mcp", help="Start the MCP server")

    args = parser.parse_args()

    if args.command == "mcp":
        from .mcp_server import main as mcp_main

        mcp_main()
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
