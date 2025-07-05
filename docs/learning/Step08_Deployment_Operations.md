# Step08: デプロイメントと運用

## 🎯 この章の目標

Docker・Kubernetes環境でのデプロイメント、CI/CD パイプライン、本番運用、スケーリング戦略を理解する

---

## 📋 概要

RAGシステムの本番運用では、コンテナ化、オーケストレーション、自動デプロイメント、スケーリング、災害復旧など、企業グレードの運用要件を満たす必要があります。Kubernetes環境での安定したサービス提供を実現します。

### 🏗️ デプロイメント・運用構成

```text
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

**注**: 現在、専用のDockerfileは未作成ですが、以下の設計方針に基づいて実装予定です。

#### Dockerコンテナ化の設計方針

```dockerfile
# マルチステージビルドの構成
# Stage 1: base - セキュリティパッチ適用済みベースイメージ
# Stage 2: dependencies - Python依存関係のビルド
# Stage 3: production - 最小限の本番用イメージ
# Stage 4: development - 開発ツール付きイメージ

FROM python:3.11-slim-bullseye AS base

# セキュリティ設定: 非rootユーザーの作成
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 必要最小限のシステムパッケージのみインストール
# 以下の最適化を実施：
# - レイヤーキャッシュの最大活用
# - 不要ファイルの即座削除
# - マルチステージでのイメージサイズ最小化
```

#### Dockerコンテナ化の設計哲学

**日本語解説**:
```
Dockerイメージ設計の原則：

1. セキュリティファースト（Security First）
   - 非rootユーザーでの実行
   - 最小権限の原則
   - 脆弱性スキャンの定期実施
   - ベースイメージの定期更新

2. イメージサイズ最適化（Size Optimization）
   - マルチステージビルド
   - 不要ファイルの除去
   - レイヤー数の最小化
   - キャッシュ効率の最大化

3. 再現性と一貫性（Reproducibility）
   - 明示的なバージョン指定
   - 環境変数の文書化
   - ビルド引数の活用
   - デターミニスティックなビルド

4. 運用効率（Operational Efficiency）
   - ヘルスチェックの組み込み
   - グレースフルシャットダウン
   - ログ出力の標準化
   - メトリクスの公開
```

### 2. Docker Compose 設定

**実装ファイル**: `../../docker-compose.yml`

#### サービス構成

このファイルでは以下の重要なサービスが定義されています：

- **rag-api**: FastAPIアプリケーション本体
- **postgres**: メタデータ管理用のPostgreSQLデータベース
- **milvus**: ベクター検索用のMilvusデータベース
- **redis**: キャッシュとCeleryのメッセージブローカー
- **etcd**: Milvusの設定管理
- **minio**: オブジェクトストレージ（Milvusのデータ保存用）

#### Docker Compose設計の特徴

```yaml
# docker-compose.yml の主要構成
version: '3.8'

services:
  # 主要サービスの定義
  # - rag-api: FastAPIアプリケーション
  # - postgres: メタデータ管理DB
  # - milvus: ベクター検索DB
  # - redis: キャッシュ・メッセージブローカー
  # - minio: オブジェクトストレージ
  # - etcd: Milvus設定管理

```

実際の`docker-compose.yml`では、以下の重要な設定が実装されています：

**サービス間の依存関係**:
- health checkによる起動順序の制御
- condition: service_healthyでの依存関係管理
- 適切なタイムアウトとリトライ設定

**ネットワーク設計**:
- 専用のbridge networkでのサービス間通信
- 固定サブネット割り当てによる安定性確保
- 外部公開ポートの最小化

**データ永続化**:
- 名前付きvolumeによるデータ永続化
- 各サービス専用のvolume設定
- バックアップ・リストア可能な構成

#### Docker Compose運用の設計哲学

**日本語解説**:
```
コンテナオーケストレーションの設計原則：

1. 宣言的な設定（Declarative Configuration）
   - YAMLによる構成管理
   - 環境変数による設定の外部化
   - バージョン管理可能な構成
   - 再現可能なデプロイメント

2. サービス間の疎結合（Loose Coupling）
   - 明確なインターフェース定義
   - 依存関係の最小化
   - 個別スケーリング可能な設計
   - 障害の局所化

3. 可観測性の確保（Observability）
   - health checkの実装
   - ログの集約設定
   - メトリクスポートの公開
   - デバッグ容易性の考慮

4. セキュリティの実装（Security Implementation）
   - 最小権限での実行
   - シークレット管理
   - ネットワーク分離
   - 脆弱性スキャンの自動化
```

---

## ☸️ Kubernetes デプロイメント

### 1. Namespace とリソース管理

Kubernetesマニフェストの検証は `../../app/deployment/kubernetes_validator.py` で実装されています。

このモジュールでは以下の重要な機能を提供：

- **マニフェスト検証**: YAML構文とKubernetesスキーマの検証
- **リソース制限チェック**: CPU/メモリの適切な制限設定確認
- **セキュリティ検証**: SecurityContext、NetworkPolicyの設定確認
- **依存関係チェック**: Service、ConfigMap、Secretの参照整合性

#### Kubernetesリソース管理の設計哲学

**日本語解説**:
```
Kubernetesデプロイメントの設計原則：

1. リソース境界の明確化（Resource Boundaries）
   - Namespaceによる論理的分離
   - ResourceQuotaによる使用量制限
   - NetworkPolicyによるネットワーク分離
   - RBACによるアクセス制御

2. 設定管理のベストプラクティス（Configuration Management）
   - ConfigMapによる設定の外部化
   - Secretによる機密情報管理
   - Helmチャートでのテンプレート化
   - GitOpsによる宣言的管理

3. 高可用性の実現（High Availability）
   - ReplicaSetでの冗長性確保
   - PodDisruptionBudgetの設定
   - ゾーン分散配置
   - ローリングアップデート戦略

4. 監視・ロギングの統合（Monitoring Integration）
   - Prometheus互換メトリクス
   - 構造化ログ出力
   - 分散トレーシング対応
   - サービスメッシュ統合
```

### 2. ConfigMap と Secret

Kubernetesでの設定管理では、以下の原則に従います：

**ConfigMap**:
- アプリケーション設定の外部化
- 環境固有の値の管理
- 設定ファイルのマウント

**Secret**:
- 機密情報の暗号化保存
- base64エンコーディング
- RBACによるアクセス制御

```yaml
# 基本的な構造例
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  # アプリケーション設定
  
---
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-system
type: Opaque
data:
  # 暗号化された機密情報
```

### 3. アプリケーションデプロイメント

**注**: Kubernetesマニフェストファイルは未作成ですが、以下の設計方針に従います：

**デプロイメントの主要設定**:
- **レプリカ数**: 3以上で高可用性確保
- **更新戦略**: RollingUpdateでゼロダウンタイム
- **ヘルスチェック**: liveness/readinessプローブ
- **リソース制限**: CPU/メモリのrequest/limit設定
- **セキュリティ**: 非rootユーザーでの実行

```yaml
# 基本構造の例
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
  template:
    spec:
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

**Service & Ingress設定**:
- **Service**: ClusterIPでの内部通信
- **Ingress**: NGINX Ingressコントローラー使用
- **TLS**: cert-managerでの自動証明書管理
- **レート制限**: DDoS対策の実装

### 4. データベース StatefulSet

**実装ファイル**: `../../app/database/production_config.py`

PostgreSQLの本番環境設定では以下の重要な機能を実装：

- **接続プール管理**: 最適なプールサイズとタイムアウト設定
- **高可用性設定**: マスター/スレーブレプリケーション対応
- **パフォーマンス最適化**: インデックス戦略とクエリ最適化
- **バックアップ/リストア**: 自動バックアップとポイントインタイムリカバリ

**StatefulSetの特徴**:
- 永続的なストレージ（PersistentVolumeClaim）
- 順序付きデプロイメント
- 安定したネットワークID
- データの永続性保証

#### データベースデプロイメントの設計哲学

**日本語解説**:
```
StatefulSetを使用したデータベース運用の原則：

1. データ永続性の保証（Data Persistence）
   - PersistentVolumeによるデータ保持
   - Pod再起動時のデータ保全
   - スナップショットによるバックアップ

2. 順序性の維持（Ordering Guarantee）
   - マスター/スレーブの適切な起動順序
   - グレースフルなシャットダウン
   - データ整合性の維持

3. ネットワークアイデンティティ（Network Identity）
   - 安定したDNS名
   - ヘッドレスサービスでの直接アクセス
   - レプリケーションの簡素化

4. 高可用性の実現（High Availability）
   - マルチレプリカ構成
   - 自動フェイルオーバー
   - 負荷分散と読み取りスケーリング
```

---

## 🔄 CI/CD パイプライン

### 1. GitHub Actions ワークフロー

現在、GitHub Actionsの設定ファイルは存在しませんが、以下のテスト自動化は実装済みです：

**実装済みテスト**:
- **単体テスト**: `../../tests/` ディレクトリ内の各テストファイル
- **統合テスト**: `../../tests/test_*_integration.py` ファイル群
- **Kubernetesマニフェスト検証**: `../../tests/test_kubernetes_manifests.py`
- **本番データベース設定テスト**: `../../tests/test_production_database.py`

#### CI/CD パイプラインの設計哲学

**日本語解説**:
```
CI/CDパイプラインの設計原則：

1. シフトレフト（Shift Left）
   - 早期の問題発見
   - 開発段階でのセキュリティチェック
   - コスト削減と品質向上
   - 自動化による人的エラー削減

2. パイプラインの段階的実行（Staged Execution）
   - 品質チェック → テスト → ビルド → デプロイ
   - 各段階での失敗時の早期終了
   - リソースの効率的利用
   - 並列実行可能なジョブの最適化

3. セキュリティの組み込み（Security Integration）
   - 依存関係の脆弱性スキャン
   - コンテナイメージスキャン
   - シークレット検出
   - SBOM（Software Bill of Materials）生成

4. 環境別デプロイメント（Environment-specific Deployment）
   - 開発 → ステージング → 本番の段階的展開
   - 環境固有の設定管理
   - ロールバック戦略の実装
   - Blue-Green/Canaryデプロイメント
```

### CI/CDパイプラインの実装

**注**: 詳細なGitHub Actionsワークフローは `.github/workflows/` ディレクトリに配置予定です。

#### パイプラインのステージ構成

1. **コード品質チェック** (`quality-check`)
   - コードフォーマット確認: `black --check app/ tests/`
   - リンティング: `ruff check app/ tests/`
   - 型チェック: `mypy app/`
   - セキュリティスキャン: `bandit -r app/`
   - 依存関係チェック: `safety check`

2. **テスト実行** (`test`)
   - 単体テスト: `pytest tests/unit/`
   - 統合テスト: `pytest tests/integration/`
   - テストサービス: PostgreSQL, Redis
   - カバレッジレポート生成

3. **セキュリティスキャン** (`security-scan`)
   - Trivyによる脆弱性スキャン
   - SARIFフォーマットでの結果出力
   - GitHub Security タブへの統合

4. **Dockerイメージビルド** (`build-image`)
   - マルチステージビルド
   - マルチプラットフォーム対応 (amd64/arm64)
   - SBOM（Software Bill of Materials）生成
   - コンテナスキャン

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
        # スモークテストは未実装（health APIテストで代用可能: ../../tests/test_health_api.py）
        echo "Running smoke tests against staging environment..."

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
        # 実際のスクリプトは未作成（以下は実装例）
        echo "Deploying image ${{ needs.build-image.outputs.image-tag }}"

    - name: Post-deployment verification

      run: |
        # 本番環境での検証テスト
        # E2Eテストは未実装（以下は実行例）
        echo "Running production readiness tests..."

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

以下はBlue-Greenデプロイメントの実装例です（実際のスクリプトファイルは未作成）：

```bash
#!/bin/bash
# scripts/blue-green-deploy.sh の例

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

監視・メトリクス収集は以下のモジュールで実装されています：

- **メトリクス収集**: `../../app/services/metrics_collection.py`
  - APIレスポンスタイム、スループット、エラー率の収集
  - 埋め込み処理のパフォーマンスメトリクス
  - 検索精度・再現率の追跡
  - システムリソース使用状況の監視

- **ログ分析**: `../../app/services/logging_analysis.py`
  - 構造化ログの収集と分析
  - エラーパターンの検出
  - セキュリティイベントの追跡
  - ログベースのメトリクス生成

- **アラート設定**: `../../app/services/alerting_service.py`
  - しきい値ベースのアラート
  - 異常検知アルゴリズム
  - エスカレーションルール
  - 通知チャネル管理

#### Prometheus監視設定

**設定ファイル**: `../../deployment/kubernetes/monitoring/prometheus-config.yaml` (未作成)

**監視対象**:
- RAG APIアプリケーションメトリクス
- PostgreSQLデータベースメトリクス
- Redisキャッシュメトリクス  
- Milvusベクトルデータベースメトリクス
- Kubernetesノードメトリクス

**アラートルール例**: `../../deployment/kubernetes/monitoring/rag_alerts.yml` (未作成)

**主要なアラートルール**:
- **APIエラー率**: 5%超過が5分間継続で警告
- **レスポンス時間**: 95パーセンタイルが2秒超過で警告  
- **検索性能**: 95パーセンタイルが5秒超過で警告
- **データベース接続**: PostgreSQL/Milvus接続失敗でクリティカル
- **リソース使用率**: メモリ90%/ディスク85%超過で警告

### 2. Grafana ダッシュボード

管理ダッシュボードの実装は `../../app/services/admin_dashboard.py` で提供されています。

このモジュールでは以下の機能を実装：

- **リアルタイムメトリクス表示**: API使用状況、検索パフォーマンス
- **システム健全性監視**: コンポーネント別のヘルスステータス
- **検索分析**: 人気クエリ、検索精度、ユーザー行動分析
- **リソース使用状況**: CPU、メモリ、ディスク、ネットワークの可視化

**Grafanaダッシュボード設定ファイル**: `../../deployment/kubernetes/monitoring/grafana-dashboard.json` (未作成)

**主要なダッシュボードパネル**:
- APIリクエスト率の推移
- エラー率のリアルタイム表示
- レスポンス時間のヒートマップ
- 検索パフォーマンスのパーセンタイル値
- データベース接続数
- メモリ使用量の推移

#### 監視システムの設計哲学

**日本語解説**:
```
監視システムの設計原則：

1. 可観測性の向上（Observability Enhancement）
   - メトリクス、ログ、トレースの統合
   - 問題の根本原因を迅速に特定
   - プロアクティブな問題検知

2. アラートの最適化（Alert Optimization）
   - 誤検知の最小化
   - 段階的な重要度設定
   - アラート疲れの防止

3. ダッシュボード設計（Dashboard Design）
   - 5秒以内の状況把握
   - 色分けによる直感的表示
   - ドリルダウン機能の提供

4. パフォーマンス影響の最小化（Performance Impact）
   - 軽量なメトリクス収集
   - サンプリングの活用
   - push/pull方式の適切な選択
```

---

## ⚖️ スケーリング戦略

### 1. Horizontal Pod Autoscaler

**設定ファイル**: `../../deployment/kubernetes/hpa.yaml` (未作成)

**スケーリング設定**:
- **最小レプリカ数**: 3
- **最大レプリカ数**: 20
- **CPU使用率**: 70%超過でスケールアウト
- **メモリ使用率**: 80%超過でスケールアウト
- **リクエストレート**: 100req/s超過でスケールアウト

**スケーリング動作**:
- **スケールアップ**: 最大50%または2ポッド増加/60秒
- **スケールダウン**: 10%ずつ減少/60秒、300秒の安定化期間

### 2. Vertical Pod Autoscaler

**設定ファイル**: `../../deployment/kubernetes/vpa.yaml` (未作成)

**リソース調整設定**:
- **更新モード**: Auto (自動的にPodを再作成してリソースを調整)
- **リソースポリシー**:
  - 最小リソース: CPU 100m, Memory 512Mi
  - 最大リソース: CPU 4000m, Memory 8Gi
  - 制御対象: CPUとメモリ

### スケーリング戦略の設計哲学

**日本語解説**:
```
オートスケーリングの設計原則：

1. 水平スケーリング（Horizontal Scaling）
   - ピーク負荷への迅速な対応
   - 負荷分散による可用性向上
   - コスト効率的なリソース利用

2. 垂直スケーリング（Vertical Scaling）
   - リソース割り当ての最適化
   - OOMエラーの防止
   - オーバープロビジョニングの削減

3. スケーリング動作の安定化（Stabilization）
   - フラッピング防止
   - 段階的なスケールアップ
   - 慈重なスケールダウン

4. メトリクスの選定（Metrics Selection）
   - CPU/メモリの基本指標
   - ビジネスメトリクスの活用
   - レスポンスタイムの考慮
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
# リソース制限の設定例
resources:
  requests:    # 最小保証リソース
    cpu: 500m
    memory: 1Gi
  limits:      # 最大使用可能リソース
    cpu: 2000m
    memory: 4Gi
```

### 3. シークレット管理

```yaml
# Kubernetes Secretを使用した機密情報管理
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
