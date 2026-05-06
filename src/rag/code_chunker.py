"""Code chunker for splitting code into semantic chunks"""

import os
import ast
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern


class CodeChunker:
    """Chunk code files into semantic units for embedding

    Supports multiple programming languages:
    - Python: by function/class/module
    - JavaScript/TypeScript: by function/class
    - Markdown: by section
    - Generic: by paragraph or fixed-size chunks
    """

    DEFAULT_CHUNK_SIZE = 500  # characters
    DEFAULT_OVERLAP = 50

    def __init__(
        self,
        chunk_size: int = None,
        overlap: int = None,
        language: str = "python",
        repo_path: str = "."
    ):
        """Initialize code chunker

        Args:
            chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks
            language: Primary language for chunking
            repo_path: Repository root path (for .gitignore)
        """
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self.overlap = overlap or self.DEFAULT_OVERLAP
        self.language = language
        self._gitignore_spec = self._load_gitignore(repo_path)

    @staticmethod
    def _load_gitignore(repo_path: str):
        """Load .gitignore patterns from repo root"""
        gitignore = Path(repo_path) / ".gitignore"
        if gitignore.exists():
            try:
                return PathSpec.from_lines(GitWildMatchPattern, gitignore.read_text().splitlines())
            except Exception:
                pass
        return None

    def _is_ignored(self, rel_path: str) -> bool:
        """Check if a relative file path matches .gitignore patterns"""
        if self._gitignore_spec is None:
            return False
        return self._gitignore_spec.match_file(rel_path)

    def chunk_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Chunk a file into semantic units

        Args:
            file_path: Path to the file

        Returns:
            List of chunk dictionaries with content and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return []

        # Detect language from file extension
        language = self._detect_language(file_path)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Failed to read file {file_path}: {e}")
            return []

        # Get relative path
        rel_path = str(file_path)

        # Chunk based on language
        if language == "python":
            chunks = self._chunk_python(content, rel_path)
        elif language in ["javascript", "typescript"]:
            chunks = self._chunk_javascript(content, rel_path, language)
        elif language == "markdown":
            chunks = self._chunk_markdown(content, rel_path)
        else:
            chunks = self._chunk_generic(content, rel_path, language)

        return chunks

    def chunk_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk all files in a directory, skipping .gitignore'd files

        Args:
            directory: Directory path
            extensions: File extensions to include
            exclude_dirs: Directories to exclude

        Returns:
            List of all chunks
        """
        directory = Path(directory).resolve()
        if not directory.exists():
            return []

        extensions = extensions or ['.py', '.js', '.ts', '.jsx', '.tsx', '.md']
        exclude_dirs = exclude_dirs or ['.git', '__pycache__', 'node_modules', 'venv', '.venv']

        all_chunks = []

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                file_path = Path(root) / file

                if file_path.suffix not in extensions:
                    continue

                # Skip gitignore'd files
                rel_path = str(file_path.relative_to(directory))
                if self._is_ignored(rel_path):
                    continue

                chunks = self.chunk_file(str(file_path))
                all_chunks.extend(chunks)

        return all_chunks

    @staticmethod
    def compute_file_hash(file_path: str) -> Optional[str]:
        """Compute SHA-256 hash of file content for change detection.

        Args:
            file_path: Path to the file

        Returns:
            Hex digest string, or None if file cannot be read
        """
        try:
            h = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
            return h.hexdigest()
        except (OSError, IOError):
            return None

    def chunk_directory_with_hashes(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Chunk all files in a directory and return chunks with file hashes,
        skipping .gitignore'd files.

        Args:
            directory: Directory path
            extensions: File extensions to include
            exclude_dirs: Directories to exclude

        Returns:
            Tuple of (all chunks, file_path -> content_hash mapping)
        """
        directory = Path(directory).resolve()
        if not directory.exists():
            return [], {}

        extensions = extensions or ['.py', '.js', '.ts', '.jsx', '.tsx', '.md']
        exclude_dirs = exclude_dirs or ['.git', '__pycache__', 'node_modules', 'venv', '.venv']

        all_chunks = []
        file_hashes = {}

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                file_path = Path(root) / file

                if file_path.suffix not in extensions:
                    continue

                rel_path = str(file_path.relative_to(directory))
                if self._is_ignored(rel_path):
                    continue

                file_str = str(file_path)
                file_hash = self.compute_file_hash(file_str)
                if file_hash is None:
                    continue

                file_hashes[file_str] = file_hash
                chunks = self.chunk_file(file_str)
                all_chunks.extend(chunks)

        return all_chunks, file_hashes

    def _detect_language(self, file_path: Path) -> str:
        """Detect language from file extension"""
        ext = file_path.suffix.lower()

        language_map = {
            # Python
            '.py': 'python',
            '.pyw': 'python',

            # JavaScript/TypeScript
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.mjs': 'javascript',
            '.cjs': 'javascript',

            # Java
            '.java': 'java',

            # C/C++
            '.c': 'c',
            '.cpp': 'c++',
            '.cc': 'c++',
            '.cxx': 'c++',
            '.h': 'c',
            '.hpp': 'c++',

            # C#
            '.cs': 'c#',

            # Go
            '.go': 'go',

            # Rust
            '.rs': 'rust',

            # Ruby
            '.rb': 'ruby',

            # PHP
            '.php': 'php',

            # Swift
            '.swift': 'swift',

            # Kotlin
            '.kt': 'kotlin',
            '.kts': 'kotlin',

            # Scala
            '.scala': 'scala',

            # Shell
            '.sh': 'shell',
            '.bash': 'shell',
            '.zsh': 'shell',

            # SQL
            '.sql': 'sql',

            # HTML/CSS
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.scss': 'css',
            '.sass': 'css',
            '.less': 'css',

            # Vue
            '.vue': 'vue',

            # Dart
            '.dart': 'dart',

            # Markdown/Text
            '.md': 'markdown',
            '.mdown': 'markdown',
            '.markdown': 'markdown',
            '.txt': 'text',

            # Config
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.xml': 'xml',

            # Lua
            '.lua': 'lua',

            # Haskell
            '.hs': 'haskell',

            # Perl
            '.pl': 'perl',
            '.pm': 'perl',
        }

        return language_map.get(ext, 'text')

    def _chunk_python(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Chunk Python code by functions and classes"""
        chunks = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fallback to generic chunking
            return self._chunk_generic(content, file_path, "python")

        # Extract module-level docstring
        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            chunks.append({
                "content": f"Module: {file_path}\n\n{module_docstring}",
                "metadata": {
                    "file_path": file_path,
                    "type": "module",
                    "language": "python"
                }
            })

        # Extract imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.append(f"import {', '.join(alias.name for alias in node.names)}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = ', '.join(alias.name for alias in node.names)
                imports.append(f"from {module} import {names}")

        if imports:
            chunks.append({
                "content": f"Imports from {file_path}:\n\n" + "\n".join(imports),
                "metadata": {
                    "file_path": file_path,
                    "type": "imports",
                    "language": "python"
                }
            })

        # Extract functions and classes
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                chunk = self._extract_class(node, content, file_path)
                chunks.append(chunk)

                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_chunk = self._extract_function(
                            item, content, file_path,
                            parent_class=node.name
                        )
                        chunks.append(method_chunk)

            elif isinstance(node, ast.FunctionDef):
                chunk = self._extract_function(node, content, file_path)
                chunks.append(chunk)

        return chunks

    def _extract_class(
        self,
        node: ast.ClassDef,
        content: str,
        file_path: str
    ) -> Dict[str, Any]:
        """Extract class definition"""
        # Get source lines
        lines = content.split('\n')
        start_line = node.lineno - 1

        # Find end line (heuristic: next class/function or EOF)
        end_line = start_line + 1
        for i in range(start_line + 1, len(lines)):
            stripped = lines[i].strip()
            if stripped.startswith(('class ', 'def ', '@')) and not lines[i].startswith('    '):
                end_line = i - 1
                break

        class_content = '\n'.join(lines[start_line:end_line + 1])

        docstring = ast.get_docstring(node) or ""

        # Build chunk content
        chunk_content = f"Class: {node.name}\n"
        chunk_content += f"File: {file_path}\n"
        chunk_content += f"Line: {node.lineno}\n\n"

        if docstring:
            chunk_content += f"Docstring:\n{docstring}\n\n"

        # Add class signature and brief summary
        chunk_content += f"Definition:\n{class_content[:self.chunk_size]}"

        return {
            "content": chunk_content,
            "metadata": {
                "file_path": file_path,
                "type": "class",
                "name": node.name,
                "line": node.lineno,
                "language": "python",
                "docstring": docstring
            }
        }

    def _extract_function(
        self,
        node: ast.FunctionDef,
        content: str,
        file_path: str,
        parent_class: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract function definition"""
        lines = content.split('\n')
        start_line = node.lineno - 1

        # Find function body extent
        indent_level = len(lines[start_line]) - len(lines[start_line].lstrip())

        end_line = start_line
        for i in range(start_line + 1, len(lines)):
            current_indent = len(lines[i]) - len(lines[i].lstrip())
            if lines[i].strip() and current_indent <= indent_level:
                end_line = i - 1
                break

        func_content = '\n'.join(lines[start_line:end_line + 1])

        docstring = ast.get_docstring(node) or ""

        # Build chunk content
        func_name = f"{parent_class}.{node.name}" if parent_class else node.name
        chunk_content = f"Function: {func_name}\n"
        chunk_content += f"File: {file_path}\n"
        chunk_content += f"Line: {node.lineno}\n\n"

        if docstring:
            chunk_content += f"Docstring:\n{docstring}\n\n"

        # Get function signature
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        if node.args.kwonlyargs:
            args.extend([arg.arg for arg in node.args.kwonlyargs])
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        signature = f"{node.name}({', '.join(args)})"
        chunk_content += f"Signature: {signature}\n\n"
        chunk_content += f"Code:\n{func_content[:self.chunk_size]}"

        return {
            "content": chunk_content,
            "metadata": {
                "file_path": file_path,
                "type": "function" if not parent_class else "method",
                "name": func_name,
                "line": node.lineno,
                "language": "python",
                "docstring": docstring,
                "signature": signature,
                "parent_class": parent_class
            }
        }

    def _chunk_javascript(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """Chunk JavaScript/TypeScript code"""
        chunks = []

        # Extract imports
        import_pattern = r'(?:import\s+.*?from\s+["\'].*?["\']|require\s*\(["\'].*?["\']\))'
        imports = re.findall(import_pattern, content)

        if imports:
            chunks.append({
                "content": f"Imports from {file_path}:\n\n" + "\n".join(imports),
                "metadata": {
                    "file_path": file_path,
                    "type": "imports",
                    "language": language
                }
            })

        # Extract functions
        func_pattern = r'(?:function\s+(\w+)\s*\([^)]*\)|const\s+(\w+)\s*=\s*(?:\([^)]*\)|[^=])*=>|(?:async\s+)?function(?:\s+\w+)?\s*\([^)]*\))'
        func_matches = re.finditer(func_pattern, content)

        for match in func_matches:
            func_name = match.group(1) or match.group(2) or "anonymous"
            start = match.start()

            # Find function body extent (heuristic)
            lines_before = content[:start].count('\n')
            lines_after = lines_before + 20  # Assume 20 lines max
            func_content = content.split('\n')[lines_before:min(lines_after, len(content.split('\n')))]
            func_content = '\n'.join(func_content[:self.chunk_size // 20])

            chunk_content = f"Function: {func_name}\n"
            chunk_content += f"File: {file_path}\n"
            chunk_content += f"Code:\n{func_content}"

            chunks.append({
                "content": chunk_content,
                "metadata": {
                    "file_path": file_path,
                    "type": "function",
                    "name": func_name,
                    "language": language
                }
            })

        # Extract classes
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?'
        class_matches = re.finditer(class_pattern, content)

        for match in class_matches:
            class_name = match.group(1)
            start = match.start()

            lines_before = content[:start].count('\n')
            lines_after = lines_before + 30
            class_content = content.split('\n')[lines_before:min(lines_after, len(content.split('\n')))]
            class_content = '\n'.join(class_content[:self.chunk_size // 20])

            chunk_content = f"Class: {class_name}\n"
            chunk_content += f"File: {file_path}\n"
            chunk_content += f"Code:\n{class_content}"

            chunks.append({
                "content": chunk_content,
                "metadata": {
                    "file_path": file_path,
                    "type": "class",
                    "name": class_name,
                    "language": language
                }
            })

        return chunks

    def _chunk_markdown(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """Chunk markdown by sections"""
        chunks = []

        # Split by headers
        sections = re.split(r'\n(?=#{1,3}\s)', content)

        for i, section in enumerate(sections):
            if not section.strip():
                continue

            # Extract header
            header_match = re.match(r'(#{1,3})\s+(.+)', section)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2)
            else:
                level = 0
                title = "Introduction" if i == 0 else f"Section {i}"

            chunk_content = f"Section: {title}\n"
            chunk_content += f"File: {file_path}\n\n"
            chunk_content += section[:self.chunk_size]

            chunks.append({
                "content": chunk_content,
                "metadata": {
                    "file_path": file_path,
                    "type": "section",
                    "title": title,
                    "level": level,
                    "language": "markdown"
                }
            })

        return chunks

    def _chunk_generic(
        self,
        content: str,
        file_path: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """Generic chunking for unknown file types"""
        chunks = []

        lines = content.split('\n')
        current_chunk = []
        current_size = 0

        for i, line in enumerate(lines):
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_content = f"File: {file_path}\n"
                chunk_content += f"Lines: {i - len(current_chunk) + 1}-{i}\n\n"
                chunk_content += '\n'.join(current_chunk)

                chunks.append({
                    "content": chunk_content,
                    "metadata": {
                        "file_path": file_path,
                        "type": "text",
                        "language": language,
                        "lines": f"{i - len(current_chunk) + 1}-{i}"
                    }
                })

                # Start new chunk with overlap
                overlap_lines = current_chunk[-self.overlap // 10:] if self.overlap > 0 else []
                current_chunk = overlap_lines
                current_size = sum(len(line) + 1 for line in overlap_lines)

            current_chunk.append(line)
            current_size += line_size

        # Save last chunk
        if current_chunk:
            start_line = len(lines) - len(current_chunk) + 1
            chunk_content = f"File: {file_path}\n"
            chunk_content += f"Lines: {start_line}-{len(lines)}\n\n"
            chunk_content += '\n'.join(current_chunk)

            chunks.append({
                "content": chunk_content,
                "metadata": {
                    "file_path": file_path,
                    "type": "text",
                    "language": language,
                    "lines": f"{start_line}-{len(lines)}"
                }
            })

        return chunks

    def get_chunk_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of chunks"""
        type_counts = {}
        language_counts = {}
        total_chars = 0

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            chunk_type = metadata.get("type", "unknown")
            language = metadata.get("language", "unknown")

            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            language_counts[language] = language_counts.get(language, 0) + 1
            total_chars += len(chunk.get("content", ""))

        return {
            "total_chunks": len(chunks),
            "total_chars": total_chars,
            "type_counts": type_counts,
            "language_counts": language_counts,
            "avg_chunk_size": total_chars // max(len(chunks), 1)
        }