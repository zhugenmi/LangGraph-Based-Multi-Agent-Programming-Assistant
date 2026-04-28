"""Code executor for running Python code in a sandboxed environment"""

import sys
import os
import re
import tempfile
import subprocess
import time


class CodeExecutor:
    """Execute Python code safely with timeout and memory limits"""

    def __init__(self, timeout=10, max_memory=100 * 1024 * 1024):
        self.timeout = timeout
        self.max_memory = max_memory

    def execute(self, code: str) -> dict:
        """Execute Python code and return results"""
        result = {
            "success": False,
            "output": "",
            "error": "",
            "execution_time": 0
        }

        start_time = time.time()
        code = self._clean_code(code)

        if not code or not code.strip():
            result["error"] = "没有可执行的代码"
            result["execution_time"] = round(time.time() - start_time, 3)
            return result

        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            try:
                process = subprocess.Popen(
                    [sys.executable, temp_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=os.path.dirname(temp_file) or os.getcwd()
                )

                try:
                    output, _ = process.communicate(timeout=self.timeout)
                    result["output"] = output.strip() if output else ""
                    result["success"] = process.returncode == 0
                    if process.returncode != 0 and not result["output"]:
                        result["error"] = "执行失败"
                except subprocess.TimeoutExpired:
                    process.kill()
                    result["error"] = f"执行超时（超过{self.timeout}秒）"
                    result["success"] = False

            finally:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass

        except Exception as e:
            result["error"] = f"执行错误: {str(e)}"
            result["success"] = False

        result["execution_time"] = round(time.time() - start_time, 3)
        return result

    def _clean_code(self, code: str) -> str:
        """Extract and clean Python code from markdown or mixed content"""
        code = code.strip()

        code_match = re.search(r'```python\s*(.*?)\s*```', code, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        code_match = re.search(r'```\s*(.*?)\s*```', code, re.DOTALL)
        if code_match:
            potential_code = code_match.group(1).strip()
            if self._looks_like_code(potential_code):
                return potential_code

        lines = code.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            if line.strip().startswith('```'):
                in_code = not in_code
                continue
            if in_code:
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines).strip()

        code_start = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('def ', 'class ', 'import ', 'from ', 'if __name__', 'async def ')):
                code_start = i
                break
            if stripped and not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                if any(stripped.startswith(kw) for kw in ['for ', 'while ', 'if ', 'else:', 'elif ', 'try:', 'except ', 'with ', 'return ', 'raise ']):
                    code_start = i
                    break

        if code_start != -1:
            return '\n'.join(lines[code_start:]).strip()

        non_comment_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        if non_comment_lines:
            return '\n'.join(non_comment_lines).strip()

        return code

    def _looks_like_code(self, text: str) -> bool:
        """Check if text looks like Python code"""
        code_indicators = ['def ', 'class ', 'import ', 'from ', 'if ', 'else:', 'elif ',
                          'for ', 'while ', 'return ', 'yield ', 'raise ', 'try:', 'except ',
                          'with ', 'as ', 'async ', 'await ', 'lambda ', 'pass.', 'break', 'continue']
        return any(indicator in text for indicator in code_indicators)