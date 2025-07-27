# Cursor IDEとの連携方法

## エラーの取得方法

### 方法1: 問題パネルからコピー
1. Cursorで「表示」→「問題」（または `Cmd+Shift+M` / `Ctrl+Shift+M`）
2. エラーを選択して右クリック→「すべてコピー」
3. テキストファイルに貼り付けて保存

### 方法2: VSCode/Cursor のコマンドラインツール
```bash
# もしCursorのCLIツールがインストールされている場合
cursor --list-extensions
cursor --inspect-extensions
```

### 方法3: ワークスペースの診断情報を取得
1. コマンドパレット（`Cmd+Shift+P` / `Ctrl+Shift+P`）を開く
2. "Developer: Show Running Extensions" を実行
3. "Developer: Capture Editor Logs" を実行

## 推奨される連携フロー

1. **エラーをファイルに保存**
   ```bash
   # Cursorの問題パネルからコピーしたエラーを保存
   pbpaste > docs/cursor_errors_$(date +%Y%m%d).txt  # Mac
   # または
   # Ctrl+V でファイルに貼り付け  # Windows/Linux
   ```

2. **エラーファイルを解析**
   ```bash
   python scripts/analyze_cursor_errors.py docs/cursor_errors_*.txt
   ```

3. **自動修正スクリプトの実行**
   ```bash
   python scripts/fix_cursor_errors.py
   ```
