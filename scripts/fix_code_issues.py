#!/usr/bin/env python3
"""
Automatically fix code quality issues detected by collect_code_issues.py
"""

import json
import subprocess  # nosec B404
import sys
from pathlib import Path


class CodeIssueFixer:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.issues_file = self.project_root / "code_issues.json"

    def load_issues(self):
        """Load issues from code_issues.json"""
        if not self.issues_file.exists():
            print("❌ code_issues.json not found. Run collect_code_issues.py first.")
            return None

        with open(self.issues_file) as f:
            return json.load(f)

    def fix_black_issues(self, issues):
        """Fix Black formatting issues"""
        if not issues:
            print("✅ No Black formatting issues")
            return True

        print(f"🔧 Fixing {len(issues)} Black formatting issues...")
        result = subprocess.run(
            ["black", "app/", "tests/"],
            capture_output=True,
            text=True,  # nosec B603,B607
        )
        if result.returncode == 0:
            print("✅ Black formatting fixed")
            return True
        else:
            print(f"❌ Black formatting failed: {result.stderr}")
            return False

    def fix_ruff_issues(self, issues):
        """Fix Ruff linting issues"""
        if not issues:
            print("✅ No Ruff linting issues")
            return True

        fixable_count = sum(1 for issue in issues if issue.get("fixable", False))
        print(f"🔧 Fixing {fixable_count}/{len(issues)} fixable Ruff issues...")

        subprocess.run(
            ["ruff", "check", "--fix", "app/", "tests/"],
            capture_output=True,
            text=True,  # nosec B603,B607
        )

        # Check remaining issues
        check_result = subprocess.run(
            ["ruff", "check", "app/", "tests/"],
            capture_output=True,
            text=True,  # nosec B603,B607
        )
        if check_result.returncode == 0:
            print("✅ All Ruff issues fixed")
            return True
        else:
            print("⚠️  Some Ruff issues remain (need manual fix)")
            # Show remaining issues
            subprocess.run(["ruff", "check", "app/", "tests/"])  # nosec B603,B607
            return False

    def show_mypy_issues(self, issues):
        """Show mypy issues (these usually need manual fixing)"""
        if not issues:
            print("✅ No mypy type checking issues")
            return True

        print(f"⚠️  {len(issues)} mypy issues need manual fixing:")
        for issue in issues:
            print(f"  - {issue['file']}:{issue['line']}: {issue['message']}")
        return False

    def show_pytest_issues(self, issues):
        """Show pytest failures (these need manual fixing)"""
        if not issues:
            print("✅ No pytest failures")
            return True

        print(f"⚠️  {len(issues)} pytest failures need manual fixing:")
        for issue in issues:
            print(f"  - {issue['test']}")
            print(f"    Error: {issue['error']}")
        return False

    def run(self):
        """Run the fixer"""
        print("🚀 Starting automatic code issue fixes...\n")

        # First, collect current issues
        print("📊 Collecting current issues...")
        result = subprocess.run(
            [sys.executable, "scripts/collect_code_issues.py"],  # nosec B603
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("❌ Failed to collect issues")
            return 1

        # Load issues
        issues = self.load_issues()
        if issues is None:
            return 1

        # Fix issues
        # Black - can be auto-fixed
        self.fix_black_issues(issues.get("black", []))

        # Ruff - some can be auto-fixed
        self.fix_ruff_issues(issues.get("ruff", []))

        # Mypy - usually needs manual fixing
        self.show_mypy_issues(issues.get("mypy", []))

        # Pytest - needs manual fixing
        self.show_pytest_issues(issues.get("pytest", []))

        # Re-run collector to verify
        print("\n📊 Re-checking for remaining issues...")
        result = subprocess.run(
            [sys.executable, "scripts/collect_code_issues.py"],  # nosec B603
            capture_output=True,
            text=True,
        )

        # Load final results
        final_issues = self.load_issues()
        if final_issues:
            total_issues = sum(len(v) for v in final_issues.values())
            if total_issues == 0:
                print("\n✨ All issues fixed! Code is ready for commit.")
                return 0
            else:
                print(
                    f"\n⚠️  {total_issues} issues remain. Manual intervention needed."
                )
                return 1

        return 0


if __name__ == "__main__":
    fixer = CodeIssueFixer()
    sys.exit(fixer.run())
