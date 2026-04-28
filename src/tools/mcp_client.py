"""MCP Client for secure access to local codebase"""

import requests
import os


class MCPClient:
    """MCP Client for interacting with MCP server"""

    def __init__(self):
        """Initialize MCP Client"""
        self.server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
        self.api_key = os.getenv("MCP_API_KEY", "")
        self.enabled = bool(self.api_key and self.server_url != "http://localhost:8000")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def read_file(self, file_path):
        """Read a file from the local codebase"""
        if not self.enabled:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        endpoint = f"{self.server_url}/api/v1/files/read"
        payload = {"file_path": file_path}
        response = requests.post(endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json().get("content")

    def write_file(self, file_path, content):
        """Write a file to the local codebase"""
        if not self.enabled:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {"success": True}

        endpoint = f"{self.server_url}/api/v1/files/write"
        payload = {"file_path": file_path, "content": content}
        response = requests.post(endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def list_files(self, directory):
        """List files in a directory"""
        if not self.enabled:
            import glob
            return glob.glob(f"{directory}/**/*", recursive=True)

        endpoint = f"{self.server_url}/api/v1/files/list"
        payload = {"directory": directory}
        response = requests.post(endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json().get("files")

    def execute_command(self, command, directory):
        """Execute a command in the specified directory"""
        if not self.enabled:
            import subprocess
            result = subprocess.run(
                command,
                shell=True,
                cwd=directory,
                capture_output=True,
                text=True
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }

        endpoint = f"{self.server_url}/api/v1/commands/execute"
        payload = {"command": command, "directory": directory}
        response = requests.post(endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()