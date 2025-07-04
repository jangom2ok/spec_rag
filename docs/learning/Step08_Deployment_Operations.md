# Step08: デプロイメントと運用

## 🎯 この章の目標
Docker・Kubernetes環境でのデプロイメント、CI/CD パイプライン、本番運用、スケーリング戦略を理解する

---

## 📋 概要

RAGシステムの本番運用では、コンテナ化、オーケストレーション、自動デプロイメント、スケーリング、災害復旧など、企業グレードの運用要件を満たす必要があります。Kubernetes環境での安定したサービス提供を実現します。

### 🏗️ デプロイメント・運用構成

```
本番環境アーキテクチャ
├── コンテナ化
│   ├── Docker Images     # アプリケーション・依存関係
│   ├── Multi-stage Build # 最適化・セキュリティ
│   ├── Base Images       # セキュリティパッチ適用
│   └── Registry管理      # イメージバージョン管理
├── Kubernetes クラスター
│   ├── Namespace分離     # 環境別（dev/staging/prod）
│   ├── Pod管理           # アプリケーション実行単位
│   ├── Service Discovery # 内部通信・ロードバランシング
│   └── Ingress Controller # 外部アクセス制御
├── データストレージ
│   ├── PostgreSQL HA     # 高可用性DB設定
│   ├── Milvus Cluster    # ベクターDB分散構成
│   ├── Redis Cluster     # キャッシュ・セッション管理
│   └── MinIO/S3          # オブジェクトストレージ
├── CI/CD パイプライン
│   ├── ソースコード管理  # Git・ブランチ戦略
│   ├── 自動テスト        # Unit/Integration/E2E
│   ├── セキュリティスキャン # 脆弱性・コンプライアンス
│   └── デプロイメント自動化 # Blue-Green・Canary
└── 運用・監視
    ├── ログ集約          # ELK Stack・Fluentd
    ├── メトリクス監視    # Prometheus・Grafana
    ├── APM              # アプリケーション性能監視
    └── アラート・通知    # PagerDuty・Slack
```

---

## 🐳 Docker コンテナ化

### 1. マルチステージ Dockerfile

```dockerfile
# ベースイメージ（セキュリティパッチ適用済み）
FROM python:3.11-slim-bullseye AS base

# セキュリティ設定
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 依存関係ビルド段階
FROM base AS dependencies

# 作業ディレクトリ設定
WORKDIR /app

# Pythonの最適化設定
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# システム依存関係インストール
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係インストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# BGE-M3 モデル事前ダウンロード（オプション）
RUN python -c "from FlagEmbedding import FlagModel; FlagModel('BAAI/BGE-M3')" || true

# 本番イメージ段階
FROM base AS production

# 作業ディレクトリ設定
WORKDIR /app

# 依存関係のコピー
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# アプリケーションコードのコピー
COPY --chown=appuser:appuser . .

# 不要ファイルの削除
RUN find . -type f -name "*.pyc" -delete \
    && find . -type d -name "__pycache__" -delete \
    && rm -rf tests/ docs/ .git/

# ヘルスチェック設定
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# セキュリティ設定
USER appuser

# ポート公開
EXPOSE 8000

# アプリケーション起動
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# 開発環境用ステージ
FROM dependencies AS development

# 開発ツールのインストール
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    ruff \
    mypy

# アプリケーションコードのコピー（開発時は差分更新）
COPY . .

# 開発サーバー起動
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### 2. Docker Compose 設定

```yaml
# docker-compose.yml
version: '3.8'

services:
  # RAG アプリケーション
  rag-api:
    build:
      context: .
      target: production
    image: rag-system:${VERSION:-latest}
    container_name: rag-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://rag_user:${DB_PASSWORD}@postgres:5432/rag_db
      - REDIS_URL=redis://redis:6379
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      milvus:
        condition: service_healthy
    volumes:
      - ./logs:/app/logs
      - /tmp:/tmp  # 一時ファイル用
    networks:
      - rag-network
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  # PostgreSQL データベース
  postgres:
    image: postgres:15-alpine
    container_name: rag-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --locale=ja_JP.UTF-8"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag_user -d rag_db"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
    networks:
      - rag-network

  # Milvus ベクターデータベース
  milvus:
    image: milvusdb/milvus:v2.3.4
    container_name: rag-milvus
    restart: unless-stopped
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: ${MINIO_PASSWORD}
    volumes:
      - milvus_data:/var/lib/milvus
      - ./milvus.yaml:/milvus/configs/milvus.yaml
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      etcd:
        condition: service_healthy
      minio:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 90s
    networks:
      - rag-network

  # Redis キャッシュ
  redis:
    image: redis:7-alpine
    container_name: rag-redis
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - rag-network

  # MinIO オブジェクトストレージ
  minio:
    image: minio/minio:latest
    container_name: rag-minio
    restart: unless-stopped
    command: server /data --console-address ":9001"
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: ${MINIO_PASSWORD}
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - rag-network

  # etcd（Milvus用）
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    container_name: rag-etcd
    restart: unless-stopped
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    volumes:
      - etcd_data:/etcd
    ports:
      - "2379:2379"
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - rag-network

volumes:
  postgres_data:
    driver: local
  milvus_data:
    driver: local
  redis_data:
    driver: local
  minio_data:
    driver: local
  etcd_data:
    driver: local

networks:
  rag-network:
    driver: bridge
    ipam:
      driver: default
      config:
        - subnet: 172.20.0.0/16
```

---

## ☸️ Kubernetes デプロイメント

### 1. Namespace とリソース管理

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system
  labels:
    name: rag-system
    environment: production
---
# resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: rag-system-quota
  namespace: rag-system
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "10"
    pods: "20"
    services: "10"
```

### 2. ConfigMap と Secret

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  database_host: "rag-postgres-service"
  database_port: "5432"
  database_name: "rag_db"
  redis_host: "rag-redis-service"
  redis_port: "6379"
  milvus_host: "rag-milvus-service"
  milvus_port: "19530"
  log_level: "INFO"
  environment: "production"
  # Milvus 設定
  milvus.yaml: |
    etcd:
      endpoints:
        - rag-etcd-service:2379
    minio:
      address: rag-minio-service
      port: 9000
      accessKeyID: minioadmin
      secretAccessKey: ${MINIO_PASSWORD}
      useSSL: false
      bucketName: "milvus-bucket"
    common:
      defaultPartitionName: "_default"
      defaultIndexName: "_default_idx"
      retentionDuration: 432000  # 5 days in seconds
    
---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-system
type: Opaque
data:
  database_password: <base64-encoded-password>
  jwt_secret_key: <base64-encoded-jwt-secret>
  minio_password: <base64-encoded-minio-password>
  api_encryption_key: <base64-encoded-encryption-key>
```

### 3. アプリケーションデプロイメント

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
  labels:
    app: rag-api
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: rag-api
        image: your-registry/rag-system:1.0.0
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DATABASE_URL
          value: "postgresql://rag_user:$(DATABASE_PASSWORD)@$(DATABASE_HOST):$(DATABASE_PORT)/$(DATABASE_NAME)"
        - name: DATABASE_HOST
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: database_host
        - name: DATABASE_PORT
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: database_port
        - name: DATABASE_NAME
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: database_name
        - name: DATABASE_PASSWORD
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: database_password
        - name: REDIS_URL
          value: "redis://$(REDIS_HOST):$(REDIS_PORT)"
        - name: REDIS_HOST
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: redis_host
        - name: REDIS_PORT
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: redis_port
        - name: MILVUS_HOST
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: milvus_host
        - name: MILVUS_PORT
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: milvus_port
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: jwt_secret_key
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: log_level
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: environment
        
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        volumeMounts:
        - name: app-logs
          mountPath: /app/logs
        - name: temp-storage
          mountPath: /tmp
        - name: milvus-config
          mountPath: /app/milvus.yaml
          subPath: milvus.yaml
      
      volumes:
      - name: app-logs
        emptyDir: {}
      - name: temp-storage
        emptyDir:
          sizeLimit: 1Gi
      - name: milvus-config
        configMap:
          name: rag-config
          items:
          - key: milvus.yaml
            path: milvus.yaml
      
      securityContext:
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        runAsNonRoot: true
      
      restartPolicy: Always
      terminationGracePeriodSeconds: 30

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
  namespace: rag-system
  labels:
    app: rag-api
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: rag-api

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-api-ingress
  namespace: rag-system
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/use-regex: "true"
    nginx.ingress.kubernetes.io/rewrite-target: /$1
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.rag-system.example.com
    secretName: rag-api-tls
  rules:
  - host: api.rag-system.example.com
    http:
      paths:
      - path: /(.*)
        pathType: Prefix
        backend:
          service:
            name: rag-api-service
            port:
              number: 80
```

### 4. データベース StatefulSet

```yaml
# postgres-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: rag-postgres
  namespace: rag-system
spec:
  serviceName: rag-postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: rag-postgres
  template:
    metadata:
      labels:
        app: rag-postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          valueFrom:
            configMapKeyRef:
              name: rag-config
              key: database_name
        - name: POSTGRES_USER
          value: "rag_user"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: database_password
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata
        ports:
        - containerPort: 5432
          name: postgres
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: postgres-config
          mountPath: /etc/postgresql/postgresql.conf
          subPath: postgresql.conf
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        livenessProbe:
          exec:
            command:
            - /bin/sh
            - -c
            - pg_isready -U rag_user -d rag_db
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          exec:
            command:
            - /bin/sh
            - -c
            - pg_isready -U rag_user -d rag_db
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
      volumes:
      - name: postgres-config
        configMap:
          name: postgres-config
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: "fast-ssd"
      resources:
        requests:
          storage: 100Gi

---
# postgres-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-postgres-service
  namespace: rag-system
spec:
  type: ClusterIP
  ports:
  - port: 5432
    targetPort: 5432
    protocol: TCP
    name: postgres
  selector:
    app: rag-postgres
```

---

## 🔄 CI/CD パイプライン

### 1. GitHub Actions ワークフロー

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  release:
    types: [published]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # 品質チェック
  quality-check:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run code formatting check
      run: black --check app/ tests/

    - name: Run linting
      run: ruff check app/ tests/

    - name: Run type checking
      run: mypy app/

    - name: Run security scan
      run: bandit -r app/

    - name: Run dependency check
      run: safety check

  # テスト実行
  test:
    runs-on: ubuntu-latest
    needs: quality-check
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Run unit tests
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379
        TESTING: true
      run: |
        pytest tests/unit/ -v --cov=app --cov-report=xml

    - name: Run integration tests
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379
        TESTING: true
      run: |
        pytest tests/integration/ -v

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
        file: ./coverage.xml
        fail_ci_if_error: true

  # セキュリティスキャン
  security-scan:
    runs-on: ubuntu-latest
    needs: quality-check
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  # Docker イメージビルド
  build-image:
    runs-on: ubuntu-latest
    needs: [test, security-scan]
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-

    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v5
      with:
        context: .
        target: production
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        platforms: linux/amd64,linux/arm64

    - name: Generate SBOM
      uses: anchore/sbom-action@v0
      with:
        image: ${{ steps.meta.outputs.tags }}
        format: spdx-json
        output-file: sbom.spdx.json

    - name: Scan image with Trivy
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ steps.meta.outputs.tags }}
        format: 'sarif'
        output: 'image-trivy-results.sarif'

  # Staging 環境デプロイ
  deploy-staging:
    runs-on: ubuntu-latest
    needs: build-image
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG_STAGING }}

    - name: Deploy to staging
      run: |
        # Helm chart または Kustomize を使用
        kubectl set image deployment/rag-api rag-api=${{ needs.build-image.outputs.image-tag }} -n rag-staging
        kubectl rollout status deployment/rag-api -n rag-staging --timeout=300s

    - name: Run smoke tests
      run: |
        # Staging環境でのスモークテスト
        python tests/smoke/test_api_health.py --base-url=https://staging-api.rag-system.example.com

  # Production 環境デプロイ
  deploy-production:
    runs-on: ubuntu-latest
    needs: [build-image, deploy-staging]
    if: github.event_name == 'release'
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG_PRODUCTION }}

    - name: Blue-Green Deployment
      run: |
        # Blue-Green デプロイメント実行
        ./scripts/blue-green-deploy.sh ${{ needs.build-image.outputs.image-tag }}

    - name: Post-deployment verification
      run: |
        # 本番環境での検証テスト
        python tests/e2e/test_production_readiness.py --base-url=https://api.rag-system.example.com

    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
        text: |
          🚀 Production deployment completed
          Version: ${{ github.event.release.tag_name }}
          Image: ${{ needs.build-image.outputs.image-tag }}
```

### 2. Blue-Green デプロイメントスクリプト

```bash
#!/bin/bash
# scripts/blue-green-deploy.sh

set -euo pipefail

IMAGE_TAG=$1
NAMESPACE="rag-system"
APP_NAME="rag-api"

echo "Starting Blue-Green deployment for $APP_NAME with image $IMAGE_TAG"

# 現在のデプロイメント色を確認
CURRENT_COLOR=$(kubectl get deployment $APP_NAME -n $NAMESPACE -o jsonpath='{.metadata.labels.color}' || echo "blue")
NEW_COLOR=$([ "$CURRENT_COLOR" = "blue" ] && echo "green" || echo "blue")

echo "Current color: $CURRENT_COLOR, New color: $NEW_COLOR"

# 新しい色のデプロイメントを作成
envsubst < deployment-template.yaml > deployment-$NEW_COLOR.yaml
kubectl apply -f deployment-$NEW_COLOR.yaml -n $NAMESPACE

# 新しいデプロイメントの準備完了を待機
echo "Waiting for new deployment to be ready..."
kubectl rollout status deployment/$APP_NAME-$NEW_COLOR -n $NAMESPACE --timeout=300s

# ヘルスチェック
echo "Running health checks..."
for i in {1..30}; do
    if kubectl run health-check-$NEW_COLOR --rm -i --restart=Never --image=curlimages/curl:latest -- \
       curl -f http://$APP_NAME-$NEW_COLOR-service/health; then
        echo "Health check passed"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo "Health check failed after 30 attempts"
        # ロールバック
        kubectl delete deployment $APP_NAME-$NEW_COLOR -n $NAMESPACE
        exit 1
    fi
    
    echo "Health check attempt $i failed, retrying..."
    sleep 10
done

# トラフィックの切り替え
echo "Switching traffic to new deployment..."
kubectl patch service $APP_NAME-service -n $NAMESPACE -p '{"spec":{"selector":{"color":"'$NEW_COLOR'"}}}'

# 暖機時間
echo "Warming up new deployment..."
sleep 30

# 最終ヘルスチェック
echo "Running final health checks..."
kubectl run final-health-check --rm -i --restart=Never --image=curlimages/curl:latest -- \
    curl -f http://$APP_NAME-service/health

# 古いデプロイメントの削除
echo "Cleaning up old deployment..."
kubectl delete deployment $APP_NAME-$CURRENT_COLOR -n $NAMESPACE --ignore-not-found=true

# メインデプロイメントのラベル更新
kubectl label deployment $APP_NAME-$NEW_COLOR -n $NAMESPACE color=$NEW_COLOR --overwrite
kubectl patch deployment $APP_NAME-$NEW_COLOR -n $NAMESPACE -p '{"metadata":{"name":"'$APP_NAME'"}}'

echo "Blue-Green deployment completed successfully!"
```

---

## 📊 運用監視システム

### 1. Prometheus 監視設定

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
      external_labels:
        cluster: 'rag-production'
        
    rule_files:
      - "rag_alerts.yml"
      
    scrape_configs:
    # RAGアプリケーション監視
    - job_name: 'rag-api'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - rag-system
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
    
    # PostgreSQL 監視
    - job_name: 'postgres-exporter'
      static_configs:
      - targets: ['postgres-exporter:9187']
    
    # Redis 監視
    - job_name: 'redis-exporter'
      static_configs:
      - targets: ['redis-exporter:9121']
    
    # Milvus 監視
    - job_name: 'milvus'
      static_configs:
      - targets: ['rag-milvus-service:9091']
      metrics_path: /metrics
    
    # Node Exporter
    - job_name: 'node-exporter'
      kubernetes_sd_configs:
      - role: node
      relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)

  rag_alerts.yml: |
    groups:
    - name: rag.rules
      rules:
      # API エラー率
      - alert: HighErrorRate
        expr: rate(http_requests_total{status_code=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 5% for 5 minutes"
      
      # レスポンス時間
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is above 2 seconds"
      
      # 検索性能
      - alert: SlowSearchPerformance
        expr: histogram_quantile(0.95, rate(search_duration_seconds_bucket[5m])) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Search performance degraded"
          description: "95th percentile search time is above 5 seconds"
      
      # データベース接続
      - alert: DatabaseConnectionFailure
        expr: up{job="postgres-exporter"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failure"
          description: "PostgreSQL database is unreachable"
      
      # Milvus 可用性
      - alert: MilvusUnavailable
        expr: up{job="milvus"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Milvus vector database unavailable"
          description: "Milvus vector database is unreachable"
      
      # メモリ使用率
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is above 90%"
      
      # ディスク使用率
      - alert: HighDiskUsage
        expr: (node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes > 0.85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High disk usage"
          description: "Disk usage is above 85%"
```

### 2. Grafana ダッシュボード

```json
{
  "dashboard": {
    "title": "RAG System Overview",
    "panels": [
      {
        "title": "API Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/second"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(http_requests_total{status_code=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100",
            "legendFormat": "Error Rate %"
          }
        ],
        "thresholds": "1,5",
        "colorBackground": true
      },
      {
        "title": "Response Time Distribution",
        "type": "heatmap",
        "targets": [
          {
            "expr": "rate(http_request_duration_seconds_bucket[5m])",
            "format": "heatmap",
            "legendFormat": "{{le}}"
          }
        ]
      },
      {
        "title": "Search Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(search_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(search_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, rate(search_duration_seconds_bucket[5m]))",
            "legendFormat": "99th percentile"
          }
        ]
      },
      {
        "title": "Database Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_stat_database_numbackends",
            "legendFormat": "Active connections"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes / 1024 / 1024",
            "legendFormat": "{{instance}} Memory (MB)"
          }
        ]
      }
    ]
  }
}
```

---

## ⚖️ スケーリング戦略

### 1. Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
```

### 2. Vertical Pod Autoscaler

```yaml
# vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: rag-api-vpa
  namespace: rag-system
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: rag-api
      minAllowed:
        cpu: 100m
        memory: 512Mi
      maxAllowed:
        cpu: 4000m
        memory: 8Gi
      controlledResources: ["cpu", "memory"]
```

---

## ❗ よくある落とし穴と対策

### 1. コンテナセキュリティ

```dockerfile
# ❌ 問題: root ユーザーで実行
USER root
CMD ["python", "app.py"]

# ✅ 対策: 非特権ユーザーで実行
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser
CMD ["python", "app.py"]
```

### 2. リソース制限不備

```yaml
# ❌ 問題: リソース制限なし
containers:
- name: rag-api
  image: rag-system:latest
  # リソース制限なし → ノード全体のリソースを消費可能

# ✅ 対策: 適切なリソース制限
containers:
- name: rag-api
  image: rag-system:latest
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 2000m
      memory: 4Gi
```

### 3. シークレット管理

```yaml
# ❌ 問題: プレーンテキストでパスワード
env:
- name: DATABASE_PASSWORD
  value: "plain-text-password"

# ✅ 対策: Kubernetes Secret使用
env:
- name: DATABASE_PASSWORD
  valueFrom:
    secretKeyRef:
      name: rag-secrets
      key: database_password
```

---

## 🎯 理解確認のための設問

### コンテナ化理解
1. マルチステージDockerfileの利点と、本番環境向けの最適化手法を3つ挙げてください
2. コンテナセキュリティで実装すべき5つのベストプラクティスを説明してください
3. ヘルスチェック（livenessProbe/readinessProbe）の違いと適切な設定値を説明してください

### Kubernetes理解
1. StatefulSetとDeploymentの違いと、データベースでStatefulSetを使う理由を説明してください
2. ConfigMapとSecretの使い分けと、機密情報管理のベストプラクティスを説明してください
3. Ingressコントローラーの役割とSSL終端の仕組みを説明してください

### CI/CD理解
1. Blue-Greenデプロイメントの利点と、実装時の注意点を説明してください
2. セキュリティスキャンをパイプラインに組み込む理由と、チェックすべき項目を説明してください
3. 段階的デプロイメント（staging → production）の重要性と検証項目を説明してください

### 運用・監視理解
1. SLI/SLO/SLAの違いと、RAGシステムで設定すべき指標を5つ挙げてください
2. オートスケーリング（HPA/VPA）の違いと、適用場面を説明してください
3. 障害対応時の優先順位と、エスカレーション基準を説明してください

---

## 📚 学習完了

全8ステップの学習が完了しました！これで以下の知識とスキルを習得できました：

### 🎓 習得したスキル
- **Step00**: システム全体アーキテクチャの理解
- **Step01**: データフローとライフサイクル管理
- **Step02**: FastAPI による REST API 設計
- **Step03**: BGE-M3 ハイブリッド検索エンジン実装
- **Step04**: 埋め込みサービスと BGE-M3 統合
- **Step05**: PostgreSQL・Milvus データモデル設計
- **Step06**: JWT・API Key 認証・認可システム
- **Step07**: エラーハンドリング・ログ・監視システム
- **Step08**: Docker・Kubernetes デプロイメント・運用

### 🚀 次の学習段階
この基礎知識をベースに、以下の高度なトピックに進むことができます：

1. **AI/ML 最適化**: モデルの微調整・量子化・推論最適化
2. **マルチモーダル対応**: 画像・音声を含む検索システム拡張
3. **分散システム**: 複数データセンター・エッジコンピューティング対応
4. **高度なセキュリティ**: ゼロトラスト・プライバシー保護技術
5. **パフォーマンス最適化**: レイテンシ削減・スループット向上

企業グレードのRAGシステム開発・運用に必要な知識は全て網羅されています。実際のプロジェクトでこれらの知識を活用し、さらなるスキル向上を目指してください！