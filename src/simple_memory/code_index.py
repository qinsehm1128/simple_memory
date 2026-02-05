"""Code indexing and parsing for Simple Memory.

Supports:
- Multi-language code parsing
- Code structure extraction (functions, classes, methods)
- Smart chunking for code files
- Incremental updates
- Batch indexing
"""

import ast
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from .config import CodeIndexConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a chunk of code for indexing."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str = "code"  # "code", "function", "class", "method", "comment"
    name: Optional[str] = None  # For functions/classes
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "name": self.name,
            **self.metadata,
        }


@dataclass
class FileInfo:
    """Information about an indexed file."""

    path: str
    hash: str
    size: int
    modified_time: float
    language: str
    chunk_count: int
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())


# Language detection based on file extension
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".vue": "vue",
    ".svelte": "svelte",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
}


def get_language(file_path: str) -> str:
    """Detect language from file extension."""
    ext = Path(file_path).suffix.lower()
    return LANGUAGE_MAP.get(ext, "unknown")


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file content."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class CodeParser:
    """Base class for code parsing."""

    def __init__(self, config: Optional[CodeIndexConfig] = None):
        self.config = config or get_config().code_index

    def parse_file(self, file_path: str, content: str) -> List[CodeChunk]:
        """Parse a file and return code chunks."""
        language = get_language(file_path)

        if language == "python" and self.config.extract_structure:
            return self._parse_python(file_path, content)
        elif language in ("javascript", "typescript") and self.config.extract_structure:
            return self._parse_js_ts(file_path, content)
        else:
            return self._parse_generic(file_path, content, language)

    def _parse_python(self, file_path: str, content: str) -> List[CodeChunk]:
        """Parse Python code and extract structure."""
        chunks = []
        lines = content.split("\n")

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fall back to generic parsing if syntax error
            return self._parse_generic(file_path, content, "python")

        # Extract module docstring
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            docstring = tree.body[0].value.value
            chunks.append(CodeChunk(
                content=f'"""{docstring}"""',
                file_path=file_path,
                start_line=1,
                end_line=tree.body[0].end_lineno or 1,
                language="python",
                chunk_type="comment",
                name="module_docstring",
            ))

        # Extract classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                start = node.lineno
                end = node.end_lineno or start
                class_content = "\n".join(lines[start - 1:end])
                chunks.append(CodeChunk(
                    content=class_content,
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                    language="python",
                    chunk_type="class",
                    name=node.name,
                    metadata={
                        "decorators": [
                            ast.unparse(d) if hasattr(ast, "unparse") else str(d)
                            for d in node.decorator_list
                        ],
                        "bases": [
                            ast.unparse(b) if hasattr(ast, "unparse") else str(b)
                            for b in node.bases
                        ],
                    },
                ))

            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Skip methods inside classes (they're included in class chunks)
                parent = self._get_parent(tree, node)
                if isinstance(parent, ast.ClassDef):
                    continue

                start = node.lineno
                end = node.end_lineno or start
                func_content = "\n".join(lines[start - 1:end])

                # Get docstring
                docstring = ast.get_docstring(node)

                chunks.append(CodeChunk(
                    content=func_content,
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                    language="python",
                    chunk_type="function",
                    name=node.name,
                    metadata={
                        "decorators": [
                            ast.unparse(d) if hasattr(ast, "unparse") else str(d)
                            for d in node.decorator_list
                        ],
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": docstring,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                    },
                ))

        # If no structure found or file is small, also add as generic chunks
        if not chunks or len(content) > self.config.chunk_size:
            generic_chunks = self._parse_generic(file_path, content, "python")
            # Merge, avoiding duplicates
            existing_ranges = {(c.start_line, c.end_line) for c in chunks}
            for gc in generic_chunks:
                if (gc.start_line, gc.end_line) not in existing_ranges:
                    chunks.append(gc)

        return chunks

    def _get_parent(self, tree: ast.AST, node: ast.AST) -> Optional[ast.AST]:
        """Get parent node in AST."""
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                if child is node:
                    return parent
        return None

    def _parse_js_ts(self, file_path: str, content: str) -> List[CodeChunk]:
        """Parse JavaScript/TypeScript code using regex patterns."""
        chunks = []
        lines = content.split("\n")
        language = get_language(file_path)

        # Patterns for JS/TS structure
        patterns = [
            # Function declarations
            (r"(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{", "function"),
            # Arrow functions assigned to const/let/var
            (r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>", "function"),
            # Class declarations
            (r"(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{", "class"),
            # Interface declarations (TS)
            (r"(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+[\w,\s]+)?\s*\{", "interface"),
            # Type declarations (TS)
            (r"(?:export\s+)?type\s+(\w+)\s*=", "type"),
        ]

        for pattern, chunk_type in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                name = match.group(1)
                start_pos = match.start()
                start_line = content[:start_pos].count("\n") + 1

                # Find the end of the block
                end_line = self._find_block_end(lines, start_line - 1)
                block_content = "\n".join(lines[start_line - 1:end_line])

                chunks.append(CodeChunk(
                    content=block_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    chunk_type=chunk_type,
                    name=name,
                ))

        # Add generic chunks for remaining content
        generic_chunks = self._parse_generic(file_path, content, language)
        existing_ranges = {(c.start_line, c.end_line) for c in chunks}
        for gc in generic_chunks:
            if (gc.start_line, gc.end_line) not in existing_ranges:
                chunks.append(gc)

        return chunks

    def _find_block_end(self, lines: List[str], start_idx: int) -> int:
        """Find the end line of a code block by matching braces."""
        brace_count = 0
        started = False

        for i in range(start_idx, len(lines)):
            line = lines[i]
            for char in line:
                if char == "{":
                    brace_count += 1
                    started = True
                elif char == "}":
                    brace_count -= 1

            if started and brace_count == 0:
                return i + 1

        return len(lines)

    def _parse_generic(self, file_path: str, content: str, language: str) -> List[CodeChunk]:
        """Generic parsing with smart chunking."""
        chunks = []
        lines = content.split("\n")

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # Calculate approximate lines per chunk
        avg_line_length = len(content) / max(len(lines), 1)
        lines_per_chunk = max(1, int(chunk_size / max(avg_line_length, 1)))
        overlap_lines = max(1, int(overlap / max(avg_line_length, 1)))

        current_start = 0
        while current_start < len(lines):
            end = min(current_start + lines_per_chunk, len(lines))

            # Try to end at a logical boundary (empty line, end of function, etc.)
            for i in range(end - 1, max(current_start + lines_per_chunk // 2, current_start), -1):
                if i < len(lines) and (
                    lines[i].strip() == ""
                    or lines[i].strip().startswith("def ")
                    or lines[i].strip().startswith("class ")
                    or lines[i].strip().startswith("function ")
                    or lines[i].strip() == "}"
                ):
                    end = i + 1
                    break

            chunk_content = "\n".join(lines[current_start:end])

            if chunk_content.strip():
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=current_start + 1,
                    end_line=end,
                    language=language,
                    chunk_type="code",
                ))

            # Move to next chunk with overlap
            current_start = end - overlap_lines
            if current_start >= len(lines) - overlap_lines:
                break

        return chunks


class CodeIndexer:
    """Main code indexing class."""

    def __init__(self, config: Optional[CodeIndexConfig] = None):
        self.config = config or get_config().code_index
        self.parser = CodeParser(self.config)
        self._indexed_files: Dict[str, FileInfo] = {}

    def should_index_file(self, file_path: str) -> bool:
        """Check if a file should be indexed."""
        path = Path(file_path)

        # Check extension
        if path.suffix.lower() not in self.config.extensions:
            return False

        # Check if in ignored directory
        for part in path.parts:
            if part in self.config.ignore_dirs:
                return False

        # Check file size
        try:
            size = path.stat().st_size
            if size > self.config.max_file_size:
                return False
        except OSError:
            return False

        return True

    def scan_directory(
        self,
        directory: str,
        recursive: bool = True,
    ) -> Generator[str, None, None]:
        """Scan a directory for indexable files."""
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return

        if recursive:
            for file_path in dir_path.rglob("*"):
                if file_path.is_file() and self.should_index_file(str(file_path)):
                    yield str(file_path)
        else:
            for file_path in dir_path.iterdir():
                if file_path.is_file() and self.should_index_file(str(file_path)):
                    yield str(file_path)

    def index_file(self, file_path: str) -> List[CodeChunk]:
        """Index a single file and return chunks."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []

        chunks = self.parser.parse_file(file_path, content)

        # Store file info
        path = Path(file_path)
        self._indexed_files[file_path] = FileInfo(
            path=file_path,
            hash=compute_file_hash(file_path),
            size=path.stat().st_size,
            modified_time=path.stat().st_mtime,
            language=get_language(file_path),
            chunk_count=len(chunks),
        )

        return chunks

    def index_directory(
        self,
        directory: str,
        recursive: bool = True,
    ) -> Generator[Tuple[str, List[CodeChunk]], None, None]:
        """Index all files in a directory."""
        for file_path in self.scan_directory(directory, recursive):
            chunks = self.index_file(file_path)
            if chunks:
                yield file_path, chunks

    def needs_reindex(self, file_path: str) -> bool:
        """Check if a file needs to be re-indexed."""
        if file_path not in self._indexed_files:
            return True

        try:
            path = Path(file_path)
            current_hash = compute_file_hash(file_path)
            stored_info = self._indexed_files[file_path]

            return (
                current_hash != stored_info.hash
                or path.stat().st_mtime > stored_info.modified_time
            )
        except OSError:
            return True

    def get_changed_files(self, directory: str) -> List[str]:
        """Get list of files that have changed since last indexing."""
        changed = []
        for file_path in self.scan_directory(directory):
            if self.needs_reindex(file_path):
                changed.append(file_path)
        return changed

    def get_indexed_files(self) -> Dict[str, FileInfo]:
        """Get information about indexed files."""
        return self._indexed_files.copy()

    def clear_index(self) -> None:
        """Clear the file index."""
        self._indexed_files.clear()


def create_code_memory_content(chunk: CodeChunk) -> str:
    """Create memory content from a code chunk for indexing."""
    parts = []

    # Add file path and language
    parts.append(f"File: {chunk.file_path}")
    parts.append(f"Language: {chunk.language}")
    parts.append(f"Lines: {chunk.start_line}-{chunk.end_line}")

    # Add type and name if available
    if chunk.chunk_type != "code":
        parts.append(f"Type: {chunk.chunk_type}")
    if chunk.name:
        parts.append(f"Name: {chunk.name}")

    # Add metadata
    if chunk.metadata:
        if chunk.metadata.get("docstring"):
            parts.append(f"Description: {chunk.metadata['docstring']}")
        if chunk.metadata.get("args"):
            parts.append(f"Arguments: {', '.join(chunk.metadata['args'])}")

    parts.append("")  # Empty line before code
    parts.append(chunk.content)

    return "\n".join(parts)


def create_code_metadata(chunk: CodeChunk) -> Dict[str, Any]:
    """Create metadata dict for a code chunk."""
    return {
        "type": "code",
        "file_path": chunk.file_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "language": chunk.language,
        "chunk_type": chunk.chunk_type,
        "name": chunk.name,
        **chunk.metadata,
    }
