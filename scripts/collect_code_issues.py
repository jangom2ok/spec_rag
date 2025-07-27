#!/usr/bin/env python3
"""
コード品質チェックツールのエラーを収集して、修正可能な形式で出力するスクリプト
"""

import json
import os
import re
import subprocess  # nosec B404
import sys
from typing import Any


class CodeIssueCollector:
    def __init__(self):
        self.issues: dict[str, list[dict[str, Any]]] = {
            "black": [],
            "ruff": [],
            "mypy": [],
            "pytest": [],
        }

    def run_black_check(self) -> list[dict[str, Any]]:
        """Blackのチェックを実行してエラーを収集"""
        print("🔍 Running Black check...")
        try:
            result = subprocess.run(
                [
                    "black",
                    "--check",
                    "--diff",
                    "app/",
                    "tests/",
                ],  # nosec B603,B607
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                # diffから変更が必要なファイルを抽出
                files = re.findall(r"--- (.+)\s+\d{4}-\d{2}-\d{2}", result.stdout)
                for file in files:
                    self.issues["black"].append(
                        {
                            "file": file,
                            "message": "Formatting required",
                            "diff": result.stdout,
                        }
                    )
        except Exception as e:
            print(f"Error running Black: {e}")
        return self.issues["black"]

    def run_ruff_check(self) -> list[dict[str, Any]]:
        """Ruffのチェックを実行してエラーを収集"""
        print("🔍 Running Ruff check...")
        try:
            result = subprocess.run(
                [
                    "ruff",
                    "check",
                    "app/",
                    "tests/",
                    "--output-format=json",
                ],  # nosec B603,B607
                capture_output=True,
                text=True,
            )
            if result.returncode != 0 and result.stdout:
                errors = json.loads(result.stdout)
                for error in errors:
                    self.issues["ruff"].append(
                        {
                            "file": error.get("filename", ""),
                            "line": error.get("location", {}).get("row", 0),
                            "column": error.get("location", {}).get("column", 0),
                            "code": error.get("code", ""),
                            "message": error.get("message", ""),
                            "fixable": error.get("fix") is not None,
                        }
                    )
        except json.JSONDecodeError:
            # JSON形式でない場合は通常の出力をパース
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if ":" in line and not line.startswith("Found"):
                    parts = line.split(":", 3)
                    if len(parts) >= 4:
                        self.issues["ruff"].append(
                            {
                                "file": parts[0],
                                "line": int(parts[1]) if parts[1].isdigit() else 0,
                                "column": int(parts[2].split()[0])
                                if parts[2].split()[0].isdigit()
                                else 0,
                                "code": parts[2].split()[1]
                                if len(parts[2].split()) > 1
                                else "",
                                "message": parts[3].strip(),
                                "fixable": False,
                            }
                        )
        except Exception as e:
            print(f"Error running Ruff: {e}")
        return self.issues["ruff"]

    def run_mypy_check(self) -> list[dict[str, Any]]:
        """mypyのチェックを実行してエラーを収集"""
        print("🔍 Running mypy check...")
        try:
            result = subprocess.run(
                [
                    "mypy",
                    "app/",
                    "--config-file",
                    "mypy.ini",
                ],  # nosec B603,B607
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    match = re.match(r"(.+):(\d+): error: (.+)", line)
                    if match:
                        self.issues["mypy"].append(
                            {
                                "file": match.group(1),
                                "line": int(match.group(2)),
                                "message": match.group(3),
                                "type": "error",
                            }
                        )
                    else:
                        match = re.match(r"(.+):(\d+): note: (.+)", line)
                        if match:
                            self.issues["mypy"].append(
                                {
                                    "file": match.group(1),
                                    "line": int(match.group(2)),
                                    "message": match.group(3),
                                    "type": "note",
                                }
                            )
        except Exception as e:
            print(f"Error running mypy: {e}")
        return self.issues["mypy"]

    def run_pytest_check(self) -> list[dict[str, Any]]:
        """pytestのチェックを実行してエラーを収集"""
        print("🔍 Running pytest check...")
        try:
            result = subprocess.run(
                [
                    "pytest",
                    "-v",
                    "--tb=short",
                    "--no-header",
                ],  # nosec B603,B607
                capture_output=True,
                text=True,
                env={**os.environ, "ENVIRONMENT": "test"},
            )
            if result.returncode != 0:
                # 失敗したテストを解析
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    if "FAILED" in line:
                        match = re.match(r"(.+) - (.+)", line)
                        if match:
                            self.issues["pytest"].append(
                                {
                                    "test": match.group(1).replace("FAILED ", ""),
                                    "error": match.group(2),
                                }
                            )
        except Exception as e:
            print(f"Error running pytest: {e}")
        return self.issues["pytest"]

    def collect_all_issues(self) -> dict[str, list[dict[str, Any]]]:
        """すべてのツールを実行してエラーを収集"""
        self.run_black_check()
        self.run_ruff_check()
        self.run_mypy_check()
        self.run_pytest_check()
        return self.issues

    def print_summary(self):
        """エラーサマリーを表示"""
        print("\n" + "=" * 60)
        print("📊 Error Summary")
        print("=" * 60)

        total_issues = 0
        for tool, issues in self.issues.items():
            count = len(issues)
            total_issues += count
            if count > 0:
                print(f"❌ {tool}: {count} issues")
            else:
                print(f"✅ {tool}: No issues")

        print(f"\nTotal issues: {total_issues}")

        # 詳細を表示
        if total_issues > 0:
            print("\n" + "=" * 60)
            print("📝 Detailed Issues")
            print("=" * 60)

            # Black issues
            if self.issues["black"]:
                print("\n🎨 Black Formatting Issues:")
                for issue in self.issues["black"]:
                    print(f"  - {issue['file']}: {issue['message']}")

            # Ruff issues
            if self.issues["ruff"]:
                print("\n🔧 Ruff Linting Issues:")
                for issue in self.issues["ruff"]:
                    fixable = "✓" if issue.get("fixable") else "✗"
                    print(
                        f"  {fixable} {issue['file']}:{issue['line']}:{issue['column']} "
                        f"{issue['code']}: {issue['message']}"
                    )

            # mypy issues
            if self.issues["mypy"]:
                print("\n📐 Mypy Type Issues:")
                for issue in self.issues["mypy"]:
                    icon = "❗" if issue["type"] == "error" else "💡"
                    print(
                        f"  {icon} {issue['file']}:{issue['line']}: {issue['message']}"
                    )

            # pytest issues
            if self.issues["pytest"]:
                print("\n🧪 Pytest Failures:")
                for issue in self.issues["pytest"]:
                    print(f"  - {issue['test']}")
                    print(f"    Error: {issue['error']}")

    def save_to_json(self, filename: str = "code_issues.json"):
        """エラーをJSONファイルに保存"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.issues, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Issues saved to {filename}")


def main():
    collector = CodeIssueCollector()
    collector.collect_all_issues()
    collector.print_summary()
    collector.save_to_json()

    # エラーがある場合は終了コード1を返す
    total_issues = sum(len(issues) for issues in collector.issues.values())
    sys.exit(1 if total_issues > 0 else 0)


if __name__ == "__main__":
    main()
