#!/usr/bin/env python3
"""
コード品質チェックスクリプト（Python版）
Black、Ruff、mypy、pytestを順番に実行し、すべてが成功するまで修正を試みる
"""

import subprocess
import sys
import time

# ANSIカラーコード
RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
NC = "\033[0m"  # No Color


def run_command(command: list[str], check: bool = True) -> tuple[int, str, str]:
    """コマンドを実行して結果を返す"""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=check)  # noqa: S603
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr


def print_status(message: str, status: str = "info"):
    """ステータスメッセージを色付きで出力"""
    if status == "error":
        print(f"{RED}{message}{NC}")
    elif status == "success":
        print(f"{GREEN}{message}{NC}")
    else:
        print(f"{YELLOW}{message}{NC}")


def check_black() -> bool:
    """Blackフォーマッティングをチェック"""
    print_status("\n1. Black フォーマッティングチェック")

    returncode, stdout, stderr = run_command(
        ["black", "--check", "--diff", "app/", "tests/"], check=False
    )

    if returncode != 0:
        print_status("Blackフォーマッティングが必要です。自動修正します...", "error")
        run_command(["black", "app/", "tests/"])
        return False
    else:
        print_status("✓ Blackフォーマッティング: OK", "success")
        return True


def check_ruff() -> bool:
    """Ruffリンティングをチェック"""
    print_status("\n2. Ruff リンティング")

    returncode, stdout, stderr = run_command(
        ["ruff", "check", "app/", "tests/"], check=False
    )

    if returncode != 0:
        print_status("Ruffエラーが見つかりました。自動修正を試みます...", "error")
        # 自動修正を試みる（エラーは無視）
        run_command(["ruff", "check", "--fix", "app/", "tests/"], check=False)
        return False
    else:
        print_status("✓ Ruff: OK", "success")
        return True


def check_mypy() -> bool:
    """mypy型チェックを実行"""
    print_status("\n3. mypy 型チェック")

    returncode, stdout, stderr = run_command(["mypy", "app/"], check=False)

    if returncode != 0:
        print_status("mypy型エラーが見つかりました", "error")
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        return False
    else:
        print_status("✓ mypy: OK", "success")
        return True


def check_pytest() -> bool:
    """pytestテストを実行"""
    print_status("\n4. pytest テスト実行")

    returncode, stdout, stderr = run_command(["pytest", "-x"], check=False)

    if returncode != 0:
        print_status("テストが失敗しました", "error")
        return False
    else:
        print_status("✓ pytest: OK", "success")
        return True


def main():
    """メイン処理"""
    print_status("=== コード品質チェックを開始します ===")

    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        print_status(f"\n試行 {attempt}/{max_attempts}")

        # すべてのチェックを実行
        checks = [check_black(), check_ruff(), check_mypy(), check_pytest()]

        # すべてのチェックが成功したか確認
        if all(checks):
            print_status("\n=== すべてのチェックが成功しました！ ===", "success")
            return 0

        # 次の試行の前に一時停止
        if attempt < max_attempts:
            print_status("\n修正を適用しました。再度チェックします...")
            time.sleep(1)

    print_status(f"\n=== {max_attempts} 回試行しましたが、エラーが残っています ===", "error")
    print_status("手動での修正が必要です", "error")
    return 1


if __name__ == "__main__":
    sys.exit(main())
