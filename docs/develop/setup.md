# 開発環境セットアップガイド

本ガイドでは、spec_ragプロジェクトの開発環境セットアップとコード品質チェックの自動化について説明します。

## 🚀 初期セットアップ

### 1. 依存関係のインストール

```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows

# 本体とデベロッパーツールのインストール
pip install -e ".[dev]"
```

### 2. Pre-commitフックの設定

**重要**: コミット前に自動でコード品質チェックを実行するため、pre-commitフックを設定してください。

```bash
# pre-commitフックのインストール
pre-commit install

# フックの動作確認
pre-commit run --all-files
```

## 🔧 コード品質チェック

### 自動実行（推奨）

#### Pre-commitフック（コミット時）

```bash
git add .
git commit -m "コミットメッセージ"
# → 自動でコード品質チェックが実行されます
```

#### 手動実行（全ファイル）

```bash
# 全ファイルに対してチェック実行
pre-commit run --all-files
```

### 手動実行（個別）

#### コードフォーマット

```bash
# Black - コードフォーマット
black app/ tests/

# フォーマットのチェックのみ
black --check app/ tests/
```

#### リンティング

```bash
# Ruff - 高速リンター
ruff check app/ tests/

# 自動修正付き
ruff check --fix app/ tests/
```

#### 型チェック

```bash
# MyPy - 型チェック
mypy app/
```

#### セキュリティチェック

```bash
# Safety - 既知の脆弱性チェック
safety check

# Bandit - セキュリティ問題検出
bandit -r app/
```

## 🧪 テスト実行

### 基本テスト

```bash
# 全テスト実行
pytest

# カバレッジ付きテスト
pytest --cov=app

# 特定のテストファイル
pytest tests/test_sample.py

# 詳細な出力
pytest -v
```

### テストマーカー

```bash
# 単体テストのみ
pytest -m unit

# 統合テストのみ
pytest -m integration

# 遅いテストを除外
pytest -m "not slow"
```

## 📊 コード品質メトリクス

### カバレッジレポート

```bash
# HTMLレポート生成
pytest --cov=app --cov-report=html
open htmlcov/index.html  # レポートを確認
```

### 品質チェック一覧

| ツール | 目的 | 実行タイミング |
|--------|------|----------------|
| **Black** | コードフォーマット | コミット時・手動 |
| **Ruff** | リンティング・import整理 | コミット時・手動 |
| **MyPy** | 型チェック | コミット時・手動 |
| **Safety** | 脆弱性チェック | CI/CD・手動 |
| **Bandit** | セキュリティ分析 | コミット時・CI/CD |
| **Pytest** | テスト実行 | CI/CD・手動 |

## 🔄 開発ワークフロー

### 推奨の開発フロー

```bash
# 1. 機能ブランチの作成
git checkout -b feature/new-feature

# 2. コード開発
# ... 開発作業 ...

# 3. 品質チェック（自動実行）
git add .
git commit -m "feat: 新機能の実装"
# → pre-commitフックが自動実行

# 4. プッシュ
git push origin feature/new-feature
# → GitHub Actions CI/CDが自動実行
```

### トラブルシューティング

#### Pre-commitフックが失敗する場合

```bash
# 自動修正を試す
pre-commit run --all-files

# 個別に修正
black app/ tests/
ruff check --fix app/ tests/

# 再コミット
git add .
git commit -m "fix: コード品質の修正"
```

#### 型チェックエラー

```bash
# 型ヒントの追加
def function_name(param: str) -> str:
    return param

# 型無視（最終手段）
# type: ignore  # コメントで無視
```

## 📝 設定ファイル

### 主要な設定ファイル

- `.pre-commit-config.yaml`: Pre-commitフック設定
- `pyproject.toml`: ツール設定（ruff, black, mypy, pytest）
- `.flake8`: Flake8設定（レガシー）
- `.github/workflows/ci.yml`: CI/CD設定

### カスタマイズ

設定の変更が必要な場合は、`pyproject.toml`の該当セクションを編集してください。

## 🎯 品質基準

### 必須要件

- [ ] **フォーマット**: Blackによる統一フォーマット
- [ ] **リンター**: Ruffエラーゼロ
- [ ] **型ヒント**: 全関数に型アノテーション
- [ ] **Docstring**: 公開API関数にGoogle形式のdocstring
- [ ] **テストカバレッジ**: 80%以上

### 推奨事項

- 関数は単一責任の原則に従う
- 複雑な処理は適切にコメントする
- テストは可読性を重視する
- エラーハンドリングを適切に行う

このガイドに従って開発環境を設定することで、高品質なコードを効率的に開発できます！

## 前提条件

- Python 3.11以上
- Docker & Docker Compose
- Git

## 1. プロジェクトのクローン

```bash
git clone <repository-url>
cd spec_rag
```

## 2. 環境変数の設定

### ローカル開発環境

ローカル開発用に `.env` ファイルを作成してください：

```bash
# Database Configuration
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Application Configuration
ENVIRONMENT=development
```

### GitHub Secrets の設定

以下のSecretsをGitHubリポジトリに設定してください：

#### テスト環境用

- `TEST_POSTGRES_USER`: テスト用PostgreSQLユーザー名
- `TEST_POSTGRES_PASSWORD`: テスト用PostgreSQLパスワード
- `TEST_POSTGRES_DB`: テスト用データベース名

#### ステージング環境用

- `STAGING_POSTGRES_USER`: ステージング用PostgreSQLユーザー名
- `STAGING_POSTGRES_PASSWORD`: ステージング用PostgreSQLパスワード
- `STAGING_POSTGRES_DB`: ステージング用データベース名

#### 本番環境用

- `PROD_POSTGRES_USER`: 本番用PostgreSQLユーザー名
- `PROD_POSTGRES_PASSWORD`: 本番用PostgreSQLパスワード
- `PROD_POSTGRES_DB`: 本番用データベース名

### GitHub Secrets の設定方法

1. GitHubリポジトリページに移動
2. `Settings` タブをクリック
3. 左サイドバーの `Secrets and variables` → `Actions` をクリック
4. `New repository secret` ボタンをクリック
5. 各Secretを追加

## 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

## 4. 開発モードでのパッケージインストール

```bash
pip install -e .
```

## 5. Docker環境の起動

```bash
docker-compose up -d
```

## 6. アプリケーションの起動

```bash
uvicorn app.main:app --reload
```

## 7. テストの実行

```bash
# 全てのテストを実行
pytest

# カバレッジ付きでテストを実行
pytest --cov=app --cov-report=term-missing
```

## 8. コード品質チェック

```bash
# フォーマットチェック
black --check app/ tests/

# リンティング
ruff check app/ tests/

# 型チェック
mypy app/
```

## 9. Pre-commitフックの設定

```bash
pre-commit install
```

## 開発サイクル

詳細な開発サイクルについては、[開発サイクル](cycle.md) を参照してください。

## セキュリティ注意事項

- `.env` ファイルは絶対にGitにコミットしないでください
- 本番環境の認証情報は必ずGitHub Secretsまたは安全な環境変数管理システムを使用してください
- パスワードや秘密鍵などの機密情報をコードに直接記述しないでください
