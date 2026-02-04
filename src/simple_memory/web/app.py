"""Flask web application for Simple Memory."""

import asyncio
import json
import logging
from functools import wraps
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from ..config import AppConfig, get_config, get_config_manager
from ..database import get_db_manager, reset_db_manager
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

        if "search" in data:
            updates["search"] = data["search"]

        if "web" in data:
            updates["web"] = data["web"]

        new_config = config_manager.update(**updates)

        # Reset memory manager to pick up new config
        reset_memory_manager()
        reset_db_manager()

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
        }

        # Test LLM
        try:
            from ..llm import get_llm_provider

            llm = get_llm_provider(config.llm)
            response = await llm.chat([{"role": "user", "content": "Say 'ok' if you can hear me."}])
            results["llm"] = {"success": True, "message": f"LLM 连接成功: {response[:50]}..."}
        except Exception as e:
            results["llm"] = {"success": False, "message": str(e)}

        # Test Embedding
        try:
            from ..embeddings import get_embedding_provider

            embedding = get_embedding_provider(config.embedding)
            vector = await embedding.embed("test")
            results["embedding"] = {
                "success": True,
                "message": f"嵌入模型连接成功。维度: {len(vector)}",
            }
        except Exception as e:
            results["embedding"] = {"success": False, "message": str(e)}

        # Core services (LLM + Embedding) must succeed
        overall_success = results["llm"]["success"] and results["embedding"]["success"]

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
            return jsonify({"success": False, "error": "系统未配置"}), 400

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
    """Add a new memory (may split into chunks if content is long)."""
    try:
        config = get_config()
        if not config.is_configured():
            return jsonify({"success": False, "error": "系统未配置"}), 400

        data = request.json
        content = data.get("content")
        user_id = data.get("user_id", "default")
        metadata = data.get("metadata", {})
        chunk_long_text = data.get("chunk_long_text", True)

        if not content:
            return jsonify({"success": False, "error": "内容不能为空"}), 400

        memory_manager = get_memory_manager()
        memory_ids = await memory_manager.add_memory(
            content=content,
            user_id=user_id,
            metadata=metadata,
            chunk_long_text=chunk_long_text,
        )

        return jsonify({
            "success": True,
            "memory_ids": memory_ids,
            "memory_id": memory_ids[0] if memory_ids else None,  # Backwards compatibility
            "chunked": len(memory_ids) > 1,
        })
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
            return jsonify({"success": False, "error": "系统未配置"}), 400

        memory_manager = get_memory_manager()
        memory = await memory_manager.get_memory(memory_id)

        if not memory:
            return jsonify({"success": False, "error": "记忆未找到"}), 404

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
            return jsonify({"success": False, "error": "系统未配置"}), 400

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
            return jsonify({"success": False, "error": "系统未配置"}), 400

        data = request.json
        query = data.get("query")
        limit = data.get("limit", 10)
        user_id = data.get("user_id")
        tags = data.get("tags")

        if not query:
            return jsonify({"success": False, "error": "查询内容不能为空"}), 400

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


# Database Management APIs


@app.route("/api/database/info", methods=["GET"])
@run_async
async def get_database_info_api():
    """Get database information and statistics."""
    try:
        db = get_db_manager()
        config = get_config()

        # Initialize if not already
        if not db._initialized:
            db.initialize(config.embedding.dimensions)

        info = db.get_database_info()
        return jsonify({"success": True, "info": info})
    except Exception as e:
        logger.exception("Error getting database info")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/database/rebuild", methods=["POST"])
@run_async
async def rebuild_database_api():
    """Rebuild the database with optional new vector dimensions.

    This will clear all vectors but preserve memory metadata.
    After rebuilding, memories need to be re-embedded.
    """
    try:
        data = request.json or {}
        new_dimensions = data.get("dimensions")
        re_embed = data.get("re_embed", False)

        config = get_config()

        # Update config with new dimensions if specified
        if new_dimensions:
            config_manager = get_config_manager()
            config_manager.update(embedding={"dimensions": new_dimensions})

        # Reset and get fresh db manager
        reset_db_manager()
        reset_memory_manager()

        db = get_db_manager()
        db.initialize(new_dimensions or config.embedding.dimensions)

        # Rebuild database
        result = db.rebuild_database(new_dimensions)

        # Re-embed all memories if requested
        if re_embed and result.get("memory_count", 0) > 0:
            memory_manager = get_memory_manager()
            await memory_manager.initialize()

            # Get all memories and re-embed them
            memories_to_reembed = db.export_memories()
            reembedded_count = 0

            for mem in memories_to_reembed:
                try:
                    # Re-embed with LLM processing
                    processed_embedding = await memory_manager.embeddings.embed(
                        mem.get("processed_content") or mem.get("content", "")
                    )
                    content_embedding = await memory_manager.embeddings.embed(
                        mem.get("content", "")
                    )

                    # Update the memory with new vectors
                    db.update_memory(mem["id"], {
                        "vector": processed_embedding,
                        "content_vector": content_embedding,
                    })
                    reembedded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to re-embed memory {mem['id']}: {e}")

            result["reembedded_count"] = reembedded_count

        return jsonify({"success": True, "result": result})
    except Exception as e:
        logger.exception("Error rebuilding database")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/database/clear", methods=["POST"])
@run_async
async def clear_database_api():
    """Completely clear the database (all memories and vectors)."""
    try:
        data = request.json or {}
        confirm = data.get("confirm", False)

        if not confirm:
            return jsonify({
                "success": False,
                "error": "请确认清空操作，设置 confirm: true"
            }), 400

        db = get_db_manager()
        config = get_config()

        if not db._initialized:
            db.initialize(config.embedding.dimensions)

        result = db.clear_database()

        # Reset managers
        reset_memory_manager()

        return jsonify({"success": True, "result": result})
    except Exception as e:
        logger.exception("Error clearing database")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/database/export", methods=["GET"])
@run_async
async def export_database_api():
    """Export all memories for backup."""
    try:
        db = get_db_manager()
        config = get_config()

        if not db._initialized:
            db.initialize(config.embedding.dimensions)

        memories = db.export_memories()
        return jsonify({
            "success": True,
            "memories": memories,
            "count": len(memories)
        })
    except Exception as e:
        logger.exception("Error exporting database")
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/api/database/import", methods=["POST"])
@run_async
async def import_database_api():
    """Import memories from backup and re-embed them."""
    try:
        data = request.json
        memories = data.get("memories", [])

        if not memories:
            return jsonify({"success": False, "error": "没有要导入的记忆"}), 400

        config = get_config()
        if not config.is_configured():
            return jsonify({"success": False, "error": "系统未配置"}), 400

        memory_manager = get_memory_manager()
        await memory_manager.initialize()

        imported_count = 0
        failed_count = 0

        for mem in memories:
            try:
                # Re-add memory (will generate new embeddings)
                await memory_manager.add_memory(
                    content=mem.get("content", ""),
                    user_id=mem.get("user_id", "default"),
                    metadata=mem.get("metadata", {}),
                    process_with_llm=True,  # Re-process with LLM
                    chunk_long_text=False,  # Don't re-chunk
                )
                imported_count += 1
            except Exception as e:
                logger.warning(f"Failed to import memory: {e}")
                failed_count += 1

        return jsonify({
            "success": True,
            "imported": imported_count,
            "failed": failed_count
        })
    except Exception as e:
        logger.exception("Error importing database")
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
