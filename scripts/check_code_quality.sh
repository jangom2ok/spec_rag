#!/bin/bash
# コード品質チェックスクリプト
# Black、Ruff、mypy、pytestを順番に実行し、すべてが成功するまで修正を試みる

set -e  # エラーが発生したら停止

# 色付きの出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== コード品質チェックを開始します ===${NC}"

# 最大試行回数
MAX_ATTEMPTS=5
attempt=1

while [ $attempt -le $MAX_ATTEMPTS ]; do
    echo -e "\n${YELLOW}試行 $attempt/$MAX_ATTEMPTS${NC}"
    
    all_passed=true
    
    # 1. Black フォーマッティング
    echo -e "\n${YELLOW}1. Black フォーマッティングチェック${NC}"
    if ! black --check --diff app/ tests/; then
        echo -e "${RED}Blackフォーマッティングが必要です。自動修正します...${NC}"
        black app/ tests/
        all_passed=false
    else
        echo -e "${GREEN}✓ Blackフォーマッティング: OK${NC}"
    fi
    
    # 2. Ruff linting
    echo -e "\n${YELLOW}2. Ruff リンティング${NC}"
    if ! ruff check app/ tests/; then
        echo -e "${RED}Ruffエラーが見つかりました。自動修正を試みます...${NC}"
        ruff check --fix app/ tests/ || true
        all_passed=false
    else
        echo -e "${GREEN}✓ Ruff: OK${NC}"
    fi
    
    # 3. mypy 型チェック
    echo -e "\n${YELLOW}3. mypy 型チェック${NC}"
    if ! mypy app/; then
        echo -e "${RED}mypy型エラーが見つかりました${NC}"
        all_passed=false
    else
        echo -e "${GREEN}✓ mypy: OK${NC}"
    fi
    
    # 4. pytest テスト
    echo -e "\n${YELLOW}4. pytest テスト実行${NC}"
    if ! pytest -x; then
        echo -e "${RED}テストが失敗しました${NC}"
        all_passed=false
    else
        echo -e "${GREEN}✓ pytest: OK${NC}"
    fi
    
    # すべてのチェックが成功した場合
    if [ "$all_passed" = true ]; then
        echo -e "\n${GREEN}=== すべてのチェックが成功しました！ ===${NC}"
        exit 0
    fi
    
    # 次の試行の前に一時停止
    if [ $attempt -lt $MAX_ATTEMPTS ]; then
        echo -e "\n${YELLOW}修正を適用しました。再度チェックします...${NC}"
        sleep 1
    fi
    
    attempt=$((attempt + 1))
done

echo -e "\n${RED}=== $MAX_ATTEMPTS 回試行しましたが、エラーが残っています ===${NC}"
echo -e "${RED}手動での修正が必要です${NC}"
exit 1