"""Tester agent for test generation and execution"""

import subprocess
import tempfile
import os
import time
from typing import Dict, Any, Optional
from src.llm.llm_model_client import get_agent_llm_client
from src.utils.helpers import build_prompt
from src.utils.prompts import TESTER_PROMPT


class Tester:
    """Agent responsible for generating tests and running them"""

    def __init__(self):
        self.client = get_agent_llm_client("tester")
        self.model = "bigmodel"

    def generate_tests(self, code: str, task_description: str) -> Dict[str, Any]:
        """Generate test cases for the given code."""
        class_name = self._extract_class_name(code)
        func_name = self._get_function_name(code)

        prompt = build_prompt(
            TESTER_PROMPT,
            task=task_description,
            code=code,
            class_name=class_name,
            function_name=func_name,
        )

        try:
            response = self.client.invoke(prompt)

            if hasattr(response, 'content'):
                test_code = response.content
            elif hasattr(response, 'text'):
                test_code = response.text
            elif isinstance(response, str):
                test_code = response
            else:
                test_code = str(response)

            import re
            test_match = re.search(r'```python\s*(.*?)\s*```', test_code, re.DOTALL)
            if test_match:
                test_code = test_match.group(1).strip()
            else:
                # Fallback: try to extract any code block
                code_match = re.search(r'```\s*(.*?)\s*```', test_code, re.DOTALL)
                if code_match:
                    extracted = code_match.group(1).strip()
                    # Only use if it contains test code indicators
                    if any(kw in extracted for kw in ['def test_', 'assert ', 'import pytest', 'unittest']):
                        test_code = extracted
                    else:
                        test_code = ""
                else:
                    test_code = ""

            if not test_code or len(test_code.strip()) < 20:
                return {
                    "success": False,
                    "error": "测试代码生成失败，LLM 返回内容不包含有效测试代码",
                    "test_code": "",
                    "test_framework": "unittest"
                }

            return {
                "success": True,
                "test_code": test_code,
                "test_framework": "unittest",
                "coverage": self._estimate_coverage(code, test_code)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "test_code": "",
                "test_framework": "unittest"
            }

    def run_tests(self, code: str, test_code: str = None) -> Dict[str, Any]:
        """Run tests on the given code."""
        original_cwd = os.getcwd()
        try:
            if not test_code or not test_code.strip():
                gen_result = self.generate_tests(code, "Test code")
                if not gen_result["success"]:
                    return gen_result
                test_code = gen_result["test_code"]

            if not test_code or len(test_code.strip()) < 20:
                return {
                    "success": False,
                    "error": "测试代码为空或无效",
                    "output": "",
                    "passed": 0,
                    "failed": 0,
                    "total": 0,
                    "duration": 0,
                }

            with tempfile.TemporaryDirectory() as temp_dir:
                main_file = os.path.join(temp_dir, "main.py")
                with open(main_file, 'w', encoding='utf-8') as f:
                    f.write(code)

                test_file = os.path.join(temp_dir, "test_main.py")
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(test_code)

                start = time.time()
                result = subprocess.run(
                    ["python", "-m", "pytest", "test_main.py", "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=temp_dir
                )
                duration = time.time() - start

                output = result.stdout + result.stderr
                passed, failed, error_info = self._parse_pytest_output(output, result.returncode)

                return {
                    "success": not error_info,
                    "passed": passed,
                    "failed": failed,
                    "total": passed + failed,
                    "output": output,
                    "error": output if error_info else None,
                    "duration": round(duration, 3),
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Test execution timed out",
                "output": "",
                "passed": 0,
                "failed": 0,
                "total": 0,
                "duration": 30,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "passed": 0,
                "failed": 0,
                "total": 0,
                "duration": 0,
            }
        finally:
            try:
                os.chdir(original_cwd)
            except OSError:
                pass

    def _parse_pytest_output(self, output: str, return_code: int = 0):
        """Parse pytest output to extract pass/fail counts and error info."""
        passed = 0
        failed = 0
        error_info = False

        # Parse the summary line: "3 passed, 1 failed in 5.2s"
        import re
        summary_match = re.search(r'(\d+)\s+passed.*?(\d+)\s+failed', output)
        if summary_match:
            passed = int(summary_match.group(1))
            failed = int(summary_match.group(2))
        else:
            # Fallback: count per-test results
            passed = len([line for line in output.split('\n') if ' PASSED' in line])
            failed = len([line for line in output.split('\n') if ' FAILED' in line])

        # Check for collection errors (no tests collected, syntax errors, etc.)
        lower_output = output.lower()
        error_patterns = ['error collecting', 'no tests ran', 'importerror',
                          'syntax error', 'fixture']
        if any(p in lower_output for p in error_patterns):
            error_info = True
        if passed == 0 and failed == 0 and return_code != 0:
            error_info = True

        return passed, failed, error_info

    def _extract_class_name(self, code: str) -> str:
        """Extract main class name from code"""
        import re
        match = re.search(r'class\s+(\w+)', code)
        return match.group(1) if match else "Code"

    def _get_function_name(self, code: str) -> str:
        """Extract main function name from code"""
        import re
        matches = re.findall(r'def\s+(\w+)', code)
        return matches[0] if matches else "main"

    def _estimate_coverage(self, code: str, test_code: str) -> float:
        """Estimate test coverage (simplified)"""
        code_lines = len([line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')])
        test_lines = len([line for line in test_code.split('\n') if line.strip() and not line.strip().startswith('#')])

        if code_lines == 0:
            return 0.0

        return min(0.8, test_lines / code_lines * 2)

    def analyze_test_results(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test results and provide feedback."""
        if not test_results.get("success"):
            return {
                "status": "failed",
                "message": "测试执行失败",
                "recommendations": [
                    "检查代码是否有语法错误",
                    "确保所有依赖已安装",
                    "查看详细的错误日志"
                ]
            }

        passed = test_results.get("passed", 0)
        failed = test_results.get("failed", 0)
        total = test_results.get("total", 0)

        if total == 0:
            return {
                "status": "warning",
                "message": "没有执行任何测试",
                "recommendations": ["检查测试代码是否正确"]
            }

        pass_rate = passed / total

        if pass_rate == 1.0:
            return {
                "status": "success",
                "message": "所有测试通过",
                "pass_rate": pass_rate,
                "recommendations": []
            }
        else:
            return {
                "status": "partial",
                "message": f"部分测试通过 ({passed}/{total})",
                "pass_rate": pass_rate,
                "failed_count": failed,
                "recommendations": [
                    "检查失败的测试用例",
                    "考虑添加边界测试",
                    "提高异常处理覆盖率"
                ]
            }
