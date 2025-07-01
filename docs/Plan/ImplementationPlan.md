# 実装計画書

## 概要

RAGシステムの**技術仕様・環境設定・運用計画書**です。技術スタック、開発環境設定、CI/CD、監視・運用手順を定義し、効率的な開発と継続的な運用を実現します。

> **関連ドキュメント**:
> - [TDD実行プラン](./TDD_実行プラン.md) - 進行管理・タスクチェックリスト
> - [開発サイクル](../Develop/開発サイクル.md) - 開発フロー・ワークフロー

## 技術スタック

### Backend

- **言語**: Python 3.11+
- **Webフレームワーク**: FastAPI 0.100+
- **非同期処理**: asyncio, celery
- **埋め込みモデル**: BAAI/BGE-M3 (HuggingFace Transformers)
- **機械学習**: PyTorch 2.0+, sentence-transformers

### Database

- **Vector Database**: Milvus 2.3+
- **Relational Database**: PostgreSQL 15+
- **Object Storage**: MinIO (S3互換)
- **Cache**: Redis 7.0+

### Infrastructure

- **Containerization**: Docker + Docker Compose
- **Orchestration**: Kubernetes (本番環境)
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)

### Development Tools

- **IDE**: VS Code, PyCharm
- **Code Quality**: ruff, black, mypy
- **Testing**: pytest, coverage
- **Documentation**: mkdocs, sphinx

## 実装スケジュール概要

> **詳細な進捗管理**: [TDD実行プラン](./TDD_実行プラン.md)を参照

### 進捗状況

- **Phase 1**: 基盤構築 (4週間)
  - ✅ 1.1 開発環境セットアップ - 完了
  - ✅ 1.2 データベース設計・構築 - 完了
  - ⏳ 1.3 基本API実装 - 次のステップ
- **Phase 2**: 埋め込み・検索機能 (6週間) - BGE-M3・ハイブリッド検索
- **Phase 3**: 高度な機能・最適化 (4週間) - Reranker・監視
- **Phase 4**: 本番環境構築・運用 (4週間) - K8s・チューニング

## プロジェクト構造

```plaintext
spec_rag/
├── app/                       # アプリケーションコード
│   ├── __init__.py
│   ├── main.py               # FastAPI エントリーポイント
│   ├── database/             # データベース管理
│   │   ├── __init__.py
│   │   └── migration.py      # マイグレーション管理
│   ├── models/               # データモデル
│   │   ├── __init__.py
│   │   ├── database.py       # SQLAlchemy モデル
│   │   └── milvus.py         # Milvus コレクション
│   ├── repositories/         # データアクセス層
│   │   ├── __init__.py
│   │   ├── document_repository.py
│   │   └── chunk_repository.py
│   ├── api/                  # API層（今後実装）
│   │   ├── __init__.py
│   │   ├── routers/
│   │   └── dependencies.py
│   ├── services/             # サービス層（今後実装）
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   └── search.py
│   └── utils/                # ユーティリティ（今後実装）
│       └── __init__.py
├── tests/                    # テストコード
│   ├── test_database_models.py
│   ├── test_milvus_collections.py
│   ├── test_migrations.py
│   ├── test_repositories.py
│   └── test_sample.py
├── docs/                     # ドキュメント
│   ├── DetailedDesign/       # 詳細設計
│   ├── Develop/              # 開発ガイド
│   ├── Plan/                 # 計画・進捗管理
│   │   ├── TDD_実行プラン.md
│   │   └── ImplementationPlan.md
│   ├── Embedding.md
│   └── README.md
├── docker-compose.yml        # 開発環境
├── requirements.txt          # Python依存関係
├── pyproject.toml           # プロジェクト設定
├── LICENSE
└── README.md
```

## 開発環境セットアップ

### 1. 必要な依存関係のインストール

```bash
# Python 3.11+ の確認
python --version

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Linux/Mac
# または
venv\Scripts\activate  # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. 開発用Docker環境の起動

```bash
# 開発用サービスの起動
docker-compose up -d

# サービスの確認
docker-compose ps
```

### 3. データベースの初期化

```bash
# データベースの作成・マイグレーション
python scripts/setup_db.py

# 初期データの投入
python scripts/load_data.py
```

## 主要な依存関係

### requirements.txt（現在の実装）

```txt
# Web Framework
fastapi[all]>=0.104.1
uvicorn[standard]>=0.24.0

# Database
sqlalchemy[asyncio]>=2.0.0
asyncpg>=0.29.0
alembic>=1.13.0

# Vector Database
pymilvus>=2.3.0

# Utilities
pydantic>=2.5.0
python-multipart>=0.0.6

# Development & Testing
pytest>=7.4.3
pytest-asyncio>=0.21.1
pytest-cov>=4.1.0
coverage>=7.3.2
ruff>=0.1.6
black>=23.11.0
mypy>=1.7.1
pre-commit>=3.5.0
safety>=2.3.5
bandit[toml]>=1.7.5

# 今後追加予定
# torch>=2.1.0                    # BGE-M3モデル用
# transformers>=4.35.2            # HuggingFace
# sentence-transformers>=2.2.2    # 埋め込みモデル
# FlagEmbedding>=1.2.2            # BGE-M3
# celery[redis]>=5.3.4            # バックグラウンドタスク
# redis>=5.0.1                    # キャッシュ
```

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "spec_rag"
version = "0.1.0"
description = "RAG System for System Development Documentation"
authors = [
    {name = "Development Team", email = "dev@example.com"},
]
dependencies = [
    "fastapi[all]>=0.104.1",
    "uvicorn[standard]>=0.24.0",
    # ... 他の依存関係
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.3",
    "pytest-asyncio>=0.21.1",
    "coverage>=7.3.2",
    "ruff>=0.1.6",
    "black>=23.11.0",
    "mypy>=1.7.1",
    "pre-commit>=3.5.0",
]

[tool.ruff]
target-version = "py311"
line-length = 88
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]

[tool.black]
line-length = 88
target-version = ["py311"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
```

## CI/CD パイプライン

### GitHub Actions (.github/workflows/ci.yml)

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e ".[dev]"

    - name: Run linting
      run: |
        ruff check src/ tests/
        black --check src/ tests/
        mypy src/

    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v4

    - name: Build Docker image
      run: |
        docker build -t spec_rag:latest .

    - name: Push to registry
      run: |
        # Docker registry への push
        echo "Pushing to registry..."
```

## 開発ガイドライン

### コーディング規約

- **PEP 8** に準拠
- **Type hints** を必須とする
- **Docstring** はGoogle形式で記述
- **テストカバレッジ** 80%以上を維持

### Git ワークフロー

```bash
# 機能開発の流れ
git checkout -b feature/search-api
# 開発作業
git add .
git commit -m "feat: ハイブリッド検索APIの実装"
git push origin feature/search-api
# Pull Request作成
```

### テスト戦略

```python
# 単体テスト例
@pytest.mark.asyncio
async def test_search_api():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post(
            "/v1/search",
            json={"query": "JWT認証", "max_results": 10}
        )
    assert response.status_code == 200
    assert len(response.json()["results"]) <= 10
```

## 監視・アラート設定

### Prometheus メトリクス

```python
# カスタムメトリクス例
from prometheus_client import Counter, Histogram, Gauge

# API呼び出し回数
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

# レスポンス時間
response_time_histogram = Histogram(
    'response_time_seconds',
    'Response time in seconds',
    ['endpoint']
)

# 処理中のドキュメント数
documents_processing_gauge = Gauge(
    'documents_processing',
    'Number of documents being processed'
)
```

### アラートルール

```yaml
# アラート設定例
groups:
  - name: rag_system_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"

      - alert: SlowResponseTime
        expr: histogram_quantile(0.95, response_time_histogram) > 1.0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Response time is too slow"
```

## 運用タスク

### 日次タスク

- [ ] システム状態の確認
- [ ] エラーログの確認
- [ ] パフォーマンスメトリクスの確認
- [ ] データバックアップの確認

### 週次タスク

- [ ] インデックスの最適化
- [ ] 検索精度の評価
- [ ] 容量使用量の確認
- [ ] セキュリティパッチの適用

### 月次タスク

- [ ] 全体的なパフォーマンス分析
- [ ] ユーザーフィードバックの分析
- [ ] システム改善計画の策定
- [ ] 災害復旧テストの実施

## まとめ

この実装計画書に基づいて、18週間でBGE-M3を使用した高精度なRAGシステムを構築できます。各フェーズでの成果物とマイルストーンを明確に定義し、継続的な改善を行うことで、システム開発における情報検索の効率を大幅に向上させることができます。
