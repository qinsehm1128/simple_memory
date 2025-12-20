"""Flask web application for Simple Memory."""

import asyncio
import json
import logging
from functools import wraps
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from ..config import AppConfig, get_config, get_config_manager
from ..memory import get_memory_manager, reset_memory_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory containing templates and static files
WEB_DIR = Path(__file__).parent
TEMPLATE_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
)
app.secret_key = "simple-memory-secret-key"


def run_async(func):
    """Decorator to run async functions in Flask."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(func(*args, **kwargs))
        finally:
            loop.close()

    return wrapper


@app.route("/")
def index():
    """Home page."""
    config = get_config()
    return render_template("index.html", config=config)


@app.route("/memories")
def memories():
    """Memories list page."""
    config = get_config()
    return render_template("memories.html", config=config)


@app.route("/settings")
def settings():
    """Settings page."""
    config = get_config()
    return render_template("settings.html", config=config)


# API Routes


@app.route("/api/config", methods=["GET"])
def get_config_api():
    """Get current configuration."""
    config = get_config()
    return jsonify(config.model_dump())


@app.route("/api/config", methods=["POST"])
def update_config_api():
    """Update configuration."""
    try:
        data = request.json
        config_manager = get_config_manager()

        # Update each section
        updates = {}

        if "llm" in data:
            updates["llm"] = data["llm"]

        if "embedding" in data:
            updates["embedding"] = data["embedding"]

        if "database" in data:
            updates["database"] = data["database"]

        if "rerank" in data:
            updates["rerank"] = data["rerank"]

        if "search" in data:
            updates["search"] = data["search"]

        if "web" in data:
            updates["web"] = data["web"]

        new_config = config_manager.update(**updates)

        # Reset memory manager to pick up new config
        reset_memory_manager()

        return jsonify({"success": True, "config": new_config.model_dump()})
    except Exception as e:
        logger.exception("Error updating config")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/config/test", methods=["POST"])
@run_async
async def test_config_api():
    """Test the current configuration."""
    try:
        config = get_config()

        results = {
            "llm": {"success": False, "message": ""},
            "embedding": {"success": False, "message": ""},
            "rerank": {"success": False, "message": "", "enabled": False},
        }

        # Test LLM
        try:
            from ..llm import get_llm_provider

            llm = get_llm_provider(config.llm)
            response = await llm.chat([{"role": "user", "content": "Say 'ok' if you can hear me."}])
            results["llm"] = {"success": True, "message": f"LLM connected: {response[:50]}..."}
        except Exception as e:
            results["llm"] = {"success": False, "message": str(e)}

        # Test Embedding
        try:
            from ..embeddings import get_embedding_provider

            embedding = get_embedding_provider(config.embedding)
            vector = await embedding.embed("test")
            results["embedding"] = {
                "success": True,
                "message": f"Embedding connected. Dimensions: {len(vector)}",
            }
        except Exception as e:
            results["embedding"] = {"success": False, "message": str(e)}

        # Test Reranker (only if enabled)
        if config.rerank.enabled:
            results["rerank"]["enabled"] = True
            try:
                from ..reranker import get_rerank_provider

                reranker = get_rerank_provider(config.rerank)
                if reranker is None:
                    results["rerank"] = {
                        "success": False,
                        "message": "Reranker not configured properly (missing API key?)",
                        "enabled": True,
                    }
                else:
                    # Test rerank with sample data (this will trigger auto-download for Ollama)
                    test_docs = ["This is a test document.", "Another test document."]
                    rerank_results = await reranker.rerank("test query", test_docs, top_k=2)
                    results["rerank"] = {
                        "success": True,
                        "message": f"Reranker connected. Provider: {config.rerank.provider}, Model: {config.rerank.model if config.rerank.provider == 'api' else config.rerank.ollama_model}",
                        "enabled": True,
                    }
            except Exception as e:
                results["rerank"] = {"success": False, "message": str(e), "enabled": True}
        else:
            results["rerank"] = {"success": True, "message": "Reranker disabled", "enabled": False}

        # Core services (LLM + Embedding) must succeed
        overall_success = results["llm"]["success"] and results["embedding"]["success"]
        # If rerank is enabled, it must also succeed
        if config.rerank.enabled:
            overall_success = overall_success and results["rerank"]["success"]

        if overall_success:
            # Mark as configured
            config_manager = get_config_manager()
            config_manager.update(configured=True)

        return jsonify({"success": overall_success, "results": results})
    except Exception as e:
        logger.exception("Error testing config")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/memories", methods=["GET"])
@run_async
async def list_memories_api():
    """List all memories."""
    try:
        config = get_config()
        if not config.is_configured():
            return jsonify({"success": False, "error": "System not configured"}), 400

        user_id = request.args.get("user_id")
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))

        memory_manager = get_memory_manager()
        memories = await memory_manager.get_all_memories(
            user_id=user_id, limit=limit, offset=offset
        )

        # Remove vector from response
        for mem in memories:
            mem.pop("vector", None)

        return jsonify({"success": True, "memories": memories})
    except Exception as e:
        logger.exception("Error listing memories")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/memories", methods=["POST"])
@run_async
async def add_memory_api():
    """Add a new memory."""
    try:
        config = get_config()
        if not config.is_configured():
            return jsonify({"success": False, "error": "System not configured"}), 400

        data = request.json
        content = data.get("content")
        user_id = data.get("user_id", "default")
        metadata = data.get("metadata", {})

        if not content:
            return jsonify({"success": False, "error": "Content is required"}), 400

        memory_manager = get_memory_manager()
        memory_id = await memory_manager.add_memory(
            content=content, user_id=user_id, metadata=metadata
        )

        return jsonify({"success": True, "memory_id": memory_id})
    except Exception as e:
        logger.exception("Error adding memory")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/memories/<memory_id>", methods=["GET"])
@run_async
async def get_memory_api(memory_id: str):
    """Get a specific memory."""
    try:
        config = get_config()
        if not config.is_configured():
            return jsonify({"success": False, "error": "System not configured"}), 400

        memory_manager = get_memory_manager()
        memory = await memory_manager.get_memory(memory_id)

        if not memory:
            return jsonify({"success": False, "error": "Memory not found"}), 404

        memory.pop("vector", None)
        return jsonify({"success": True, "memory": memory})
    except Exception as e:
        logger.exception("Error getting memory")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/memories/<memory_id>", methods=["DELETE"])
@run_async
async def delete_memory_api(memory_id: str):
    """Delete a memory."""
    try:
        config = get_config()
        if not config.is_configured():
            return jsonify({"success": False, "error": "System not configured"}), 400

        memory_manager = get_memory_manager()
        success = await memory_manager.delete_memory(memory_id)

        return jsonify({"success": success})
    except Exception as e:
        logger.exception("Error deleting memory")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/memories/search", methods=["POST"])
@run_async
async def search_memories_api():
    """Search memories."""
    try:
        config = get_config()
        if not config.is_configured():
            return jsonify({"success": False, "error": "System not configured"}), 400

        data = request.json
        query = data.get("query")
        limit = data.get("limit", 10)
        user_id = data.get("user_id")
        tags = data.get("tags")

        if not query:
            return jsonify({"success": False, "error": "Query is required"}), 400

        memory_manager = get_memory_manager()
        results = await memory_manager.search(
            query=query, limit=limit, user_id=user_id, tags=tags
        )

        # Remove vector from response
        for mem in results:
            mem.pop("vector", None)

        return jsonify({"success": True, "results": results})
    except Exception as e:
        logger.exception("Error searching memories")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/stats", methods=["GET"])
@run_async
async def get_stats_api():
    """Get memory statistics."""
    try:
        config = get_config()
        if not config.is_configured():
            return jsonify(
                {"success": True, "stats": {"total_memories": 0, "tags": [], "configured": False}}
            )

        memory_manager = get_memory_manager()
        count = await memory_manager.count_memories()
        tags = await memory_manager.get_all_tags()

        return jsonify(
            {
                "success": True,
                "stats": {"total_memories": count, "tags": tags, "configured": True},
            }
        )
    except Exception as e:
        logger.exception("Error getting stats")
        return jsonify({"success": False, "error": str(e)}), 400


def main():
    """Run the web server."""
    config = get_config()
    app.run(
        host=config.web.host,
        port=config.web.port,
        debug=config.web.debug,
    )


if __name__ == "__main__":
    main()
