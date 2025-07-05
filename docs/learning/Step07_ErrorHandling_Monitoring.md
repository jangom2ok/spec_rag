# Step07: エラーハンドリングと監視

## 🎯 この章の目標

構造化エラーハンドリング、ログ管理、メトリクス収集、アラート設定、トラブルシューティング手法を理解する

---

## 📋 概要

本番環境でのRAGシステムでは、様々な障害や例外が発生します。適切なエラーハンドリングと包括的な監視により、システムの可用性と信頼性を確保し、問題の早期発見・迅速な対応を実現します。

### 🏗️ 監視・エラーハンドリング構成

```text
監視・エラーハンドリング システム
├── エラーハンドリング
│   ├── 例外分類         # ビジネス例外・システム例外
│   ├── 構造化レスポンス # 統一エラーフォーマット
│   ├── 回復処理         # リトライ・フォールバック
│   └── エラー追跡       # スタックトレース・コンテキスト
├── ログ管理
│   ├── 構造化ログ       # JSON形式・メタデータ付与
│   ├── ログレベル       # DEBUG/INFO/WARN/ERROR/CRITICAL
│   ├── ログローテーション # サイズ・日時ベース回転
│   └── セキュリティログ # 認証・認可・セキュリティイベント
├── メトリクス収集
│   ├── システムメトリクス # CPU/メモリ/ディスク
│   ├── アプリメトリクス   # レスポンス時間・スループット
│   ├── ビジネスメトリクス # 検索精度・ユーザー満足度
│   └── カスタムメトリクス # RAG固有の指標
└── アラート・通知
    ├── 閾値ベースアラート # CPU/メモリ使用率
    ├── 異常検知アラート   # 機械学習ベース
    ├── エラー率アラート   # エラー発生率監視
    └── 外部通知           # Slack/Teams/Email
```

---

## 🚨 エラーハンドリング システム

### 1. 例外分類と階層化

**実装ファイル**: `../../app/core/exceptions.py`

#### カスタム例外の設計思想

本システムでは、エラーを体系的に管理するため、階層化されたカスタム例外クラスを実装しています。

**エラーカテゴリの分類**:

- **VALIDATION**: 入力値検証エラー（HTTPステータス400）
- **AUTHENTICATION**: 認証失敗（HTTPステータス401）
- **AUTHORIZATION**: 認可失敗（HTTPステータス403）
- **BUSINESS_LOGIC**: ビジネスルール違反
- **EXTERNAL_SERVICE**: 外部サービス障害（HTTPステータス503）
- **SYSTEM_ERROR**: システム内部エラー
- **RESOURCE_ERROR**: リソース枯渇（HTTPステータス507）

**エラー重要度の定義**:

- **LOW**: 通常の業務フローで発生する軽微なエラー
- **MEDIUM**: 注意が必要だが、システム全体には影響しない
- **HIGH**: 重要な機能に影響があり、早急な対応が必要
- **CRITICAL**: システム全体に影響し、即座の対応が必須

#### 例外クラスの特徴

1. **統一されたエラー形式**: すべての例外は`to_dict()`メソッドで標準化されたJSON形式に変換可能
2. **詳細情報の保持**: `details`フィールドに文脈固有の情報を格納
3. **原因追跡**: `cause`フィールドで例外の連鎖を追跡可能
4. **HTTPステータスの自動マッピング**: 各例外クラスが適切なHTTPステータスコードを返却

### 2. 統一エラーハンドラー

**実装ファイル**: `../../app/main.py` (エラーハンドラー設定)

#### エラーハンドリングの統一アプローチ

本システムでは、すべてのエラーを一元的に処理し、一貫した形式でクライアントに返却します。

**エラーハンドラーの主要機能**:

1. **リクエストID生成**: すべてのエラーに一意のIDを付与し、ログとの関連付けを容易に
2. **エラーログ記録**: 構造化ログとして詳細情報を記録
3. **メトリクス収集**: エラー発生率や種類を監視システムに送信
4. **アラート判定**: 重要度に応じて通知を発送
5. **レスポンス生成**: クライアント向けの標準化されたエラーレスポンス

#### エラー種別ごとの処理

**カスタム例外の処理**:

- ビジネスロジック例外は詳細情報を含めてクライアントに返却
- システム例外は内部詳細を隠蔽し、一般的なメッセージのみ返却

**バリデーションエラーの処理**:

- FastAPIのRequestValidationErrorをキャッチ
- フィールドごとの詳細なエラー情報を構造化して返却
- HTTPステータス422（Unprocessable Entity）を使用

**予期しない例外の処理**:

- スタックトレースを含む詳細ログを記録
- 緊急アラートを発報
- クライアントには最小限の情報のみ返却（セキュリティ考慮）

#### セキュリティ考慮事項

- 本番環境では内部エラーの詳細を隠蔽
- SQLインジェクションなどの攻撃痕跡をログに記録
- 機密情報（パスワード、トークン）はログから除外

### 3. リトライ・回復処理

**実装ファイル**: `../../app/core/retry.py`

#### リトライメカニズム

外部サービスとの通信では、一時的な障害に対してリトライ処理を実装しています。

**リトライ戦略**:

1. **指数バックオフ**: 失敗するたびに待機時間を指数関数的に増加
   - 初回: 1秒待機
   - 2回目: 2秒待機
   - 3回目: 4秒待機
   - 最大待機時間: 60秒

2. **ジッター追加**: 同時リトライによる負荷集中を防止
   - 待機時間に0.5〜1.0倍のランダムな係数を乗算

3. **リトライ対象の限定**: 特定の例外のみリトライ
   - ネットワークエラー
   - タイムアウト
   - 503 Service Unavailable

#### サーキットブレーカーパターン

連続的な失敗を検知し、システムを保護するメカニズムです。

**状態遷移**:

1. **CLOSED（閉）**: 正常状態、すべてのリクエストを通過
2. **OPEN（開）**: 障害状態、すべてのリクエストを即座に失敗
3. **HALF_OPEN（半開）**: 回復試行状態、限定的なリクエストで回復を確認

**設定パラメータ**:

- **失敗閾値**: 5回連続失敗でサーキットを開く
- **回復タイムアウト**: 60秒後に回復試行
- **対象例外**: 外部サービスエラーのみ対象

#### 適用箇所

主に以下のサービスで使用：

- Milvusベクターデータベースへの接続
- 埋め込みモデルAPIの呼び出し
- 外部認証サービスとの通信

---

## 📊 ログ管理システム

### 1. 構造化ログ設定

**実装ファイル**: `../../app/services/logging_analysis.py`

#### ログ管理システムの設計

本システムでは、構造化ログを使用してシステムの動作を詳細に記録・分析します。

```python
# 構造化ログの基本フォーマット
{
    "timestamp": "2024-01-01T00:00:00Z",
    "level": "INFO",
    "service": "rag-system",
    "component": "search_engine",
    "message": "Search query processed",
    "request_id": "unique-request-id",
    "user_id": "user-123",
    "metadata": {
        "query": "example search",
        "results_count": 10,
        "processing_time_ms": 150
    }
}
```

#### ログレベルの使い分け

```python
# ログレベルの定義と使用基準
LOG_LEVELS = {
    "DEBUG": "開発時の詳細情報、本番では無効化",
    "INFO": "正常な動作の記録、監査ログ",
    "WARNING": "異常ではないが注意が必要な事象",
    "ERROR": "エラーが発生したが処理は継続可能",
    "CRITICAL": "システム停止につながる重大なエラー"
}
```

#### ログ分析機能の実装

`logging_analysis.py`では以下の機能を提供：

1. **パターン検出**: エラーパターンの自動検出
2. **統計分析**: ログからのメトリクス生成
3. **異常検知**: 通常と異なるログパターンの検出
4. **相関分析**: 複数のログイベントの関連性分析

#### ログ管理の設計哲学（構造化ログの思想）

**日本語解説**:
```
構造化ログの設計原則：

1. 機械可読性（Machine Readability）
   - JSON形式による構造化データ
   - 自動解析・集計が容易
   - ログ分析ツールとの統合

2. コンテキストの保持（Context Preservation）
   - request_idによるトランザクション追跡
   - ユーザー情報、操作内容の記録
   - エラー発生時の状況再現が可能

3. セキュリティとプライバシー（Security & Privacy）
   - 機密情報の自動マスキング
   - PII（個人識別情報）の適切な処理
   - 監査要件への準拠

4. パフォーマンス考慮（Performance Consideration）
   - 非同期ログ出力
   - バッファリングによる効率化
   - ログレベルによる出力制御
```

```python
# 構造化ログの実装例
class StructuredLogger:
    """構造化ログ管理"""
    
    # JSONフォーマッターによる構造化ログ出力
    # タイムスタンプ、ログレベル、サービス情報を自動付与
    # 機密データの自動マスキング機能
    # ファイルハンドラーによるエラーログの永続化
```

### 2. セキュリティ監査ログ

```python
class SecurityAuditLogger:
    """セキュリティ監査ログ"""

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    async def log_authentication_attempt(
        self,
        email: str,
        success: bool,
        ip_address: str,
        user_agent: str,
        failure_reason: str = None
    ):
        """認証試行ログ"""

        self.logger.info(
            "Authentication attempt",
            event_type="authentication",
            email=email,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            failure_reason=failure_reason,
            security_event=True
        )

    async def log_authorization_failure(
        self,
        user_id: str,
        resource: str,
        action: str,
        ip_address: str
    ):
        """認可失敗ログ"""

        self.logger.warning(
            "Authorization failed",
            event_type="authorization_failure",
            user_id=user_id,
            resource=resource,
            action=action,
            ip_address=ip_address,
            security_event=True
        )

    async def log_suspicious_activity(
        self,
        user_id: str,
        activity_type: str,
        details: dict,
        risk_score: float,
        ip_address: str
    ):
        """疑わしい活動ログ"""

        log_level = "critical" if risk_score > 0.8 else "warning"

        getattr(self.logger, log_level)(
            "Suspicious activity detected",
            event_type="suspicious_activity",
            user_id=user_id,
            activity_type=activity_type,
            risk_score=risk_score,
            ip_address=ip_address,
            details=details,
            security_event=True
        )

    async def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        sensitive_data: bool = False
    ):
        """データアクセスログ"""

        self.logger.info(
            "Data access",
            event_type="data_access",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            sensitive_data=sensitive_data,
            audit_event=True
        )

    async def log_system_change(
        self,
        user_id: str,
        change_type: str,
        old_value: Any,
        new_value: Any,
        resource: str
    ):
        """システム変更ログ"""

        self.logger.info(
            "System configuration changed",
            event_type="system_change",
            user_id=user_id,
            change_type=change_type,
            old_value=str(old_value),
            new_value=str(new_value),
            resource=resource,
            audit_event=True
        )
```

---

## 📈 メトリクス収集システム

### 1. Prometheus メトリクス

**実装ファイル**: `../../app/services/metrics_collector.py`

#### メトリクス収集の設計思想

Prometheusフォーマットに準拠したメトリクスを収集し、システムの健全性とパフォーマンスを可視化します。

**メトリクスの種類**:

1. **Counter（カウンター）**: 単調増加する累積値
   - HTTPリクエスト総数
   - エラー発生件数
   - 検索実行回数

2. **Histogram（ヒストグラム）**: 分布を記録
   - レスポンスタイム分布
   - 検索処理時間
   - 埋め込み生成時間

3. **Gauge（ゲージ）**: 増減する瞬間値
   - アクティブな接続数
   - キャッシュヒット率
   - キューサイズ

4. **Info（情報）**: システムのメタデータ
   - バージョン情報
   - 起動パラメータ
   - 環境情報

#### 主要なメトリクス

**パフォーマンス関連**:

- `http_request_duration_seconds`: APIレスポンスタイム
- `search_duration_seconds`: 検索処理時間
- `embedding_duration_seconds`: ベクトル生成時間

**可用性関連**:

- `http_requests_total`: リクエスト総数（ステータスコード別）
- `errors_total`: エラー発生数（カテゴリ・重要度別）
- `active_connections`: 同時接続数

**リソース使用率**:

- `cache_hit_ratio`: キャッシュ効率
- `queue_size`: 処理待ちタスク数
- `memory_usage_bytes`: メモリ使用量

#### バケット設定の最適化

ヒストグラムのバケット境界値は、SLO（Service Level Objective）に基づいて設定：

- **APIレスポンス**: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]秒
- **検索処理**: [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]秒
- **埋め込み生成**: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]秒

これにより、95パーセンタイル値などの統計情報を正確に把握可能。

### 2. カスタムメトリクス

**実装ファイル**: `../../app/services/rag_metrics.py`

#### RAGシステム固有のメトリクス

検索精度とユーザー体験を定量的に評価するため、RAGシステム専用のメトリクスを実装しています。

**品質指標**:

1. **検索精度（Precision）**:
   - 返却された結果のうち、関連性の高い結果の割合
   - 上位5件でのクリック率を基に算出
   - 検索タイプ別（hybrid/semantic/keyword）に測定

2. **検索再現率（Recall）**:
   - 存在する関連文書のうち、実際に検索できた割合
   - テストセットとの比較により算出
   - 検索漏れの発見に有効

3. **ユーザー満足度**:
   - ユーザーフィードバックの平均評価
   - 5段階評価で収集
   - 24時間/7日間/30日間の時間窓で集計

**システム健全性指標**:

1. **ドキュメント鮮度**:
   - 各ソースタイプ別の平均文書更新経過日数
   - 古い情報の検出に使用
   - 更新優先度の決定に活用

2. **埋め込みカバレッジ**:
   - 全文書に対するベクトル生成済み文書の割合
   - 検索可能な文書の網羅性を示す
   - 90%以上を目標値として設定

#### メトリクス計算の実装

**定期実行タスク**:

- 1時間ごと：基本的な品質メトリクス更新
- 日次：詳細な分析レポート生成
- 週次：トレンド分析と異常検知

**データソース**:

- 検索ログ：ユーザーの検索クエリと結果
- クリックログ：ユーザーが選択した結果
- フィードバック：明示的な評価データ
- システムログ：処理時間やエラー情報

**活用方法**:

1. ダッシュボードでのリアルタイム監視
2. 検索アルゴリズムの改善指標
3. A/Bテストでの効果測定
4. SLO達成状況の追跡

---

## 🚨 アラート・通知システム

### 1. アラート設定

**実装ファイル**: `../../app/services/alert_manager.py`

#### アラートシステムの設計思想

閾値ベースの監視により、システムの異常を早期に検出し、適切な担当者に通知します。

**アラートの重要度**:

1. **INFO（情報）**:
   - 正常な範囲内での注目すべき事象
   - 例：デプロイ完了、定期メンテナンス開始
   - 通知：ログ記録のみ

2. **WARNING（警告）**:
   - 注意が必要だが、即座の対応は不要
   - 例：CPU使用率70%超過、エラー率1%超過
   - 通知：Slackチャンネル、メール

3. **CRITICAL（緊急）**:
   - 即座の対応が必要な重大な問題
   - 例：サービス停止、データベース接続エラー
   - 通知：PagerDuty、電話通知、全チャンネル

#### アラートルールの構成要素

**基本パラメータ**:

- **メトリクス名**: 監視対象の指標
- **条件式**: 閾値との比較演算子（>, <, >=, <=, ==）
- **閾値**: アラート発火の境界値
- **継続時間**: 条件が継続すべき時間（誤検知防止）

**高度な設定**:

- **ラベルフィルタ**: 特定の条件下でのみ評価
- **時間帯制限**: 営業時間内のみ通知など
- **エスカレーション**: 未対応時の上位者への通知

#### 主要なアラートルール例

1. **パフォーマンス監視**:
   - API応答時間 > 2秒が5分間継続 → WARNING
   - API応答時間 > 5秒が1分間継続 → CRITICAL

2. **可用性監視**:
   - エラー率 > 1%が10分間継続 → WARNING
   - エラー率 > 5%が1分間継続 → CRITICAL

3. **リソース監視**:
   - メモリ使用率 > 80%が10分間継続 → WARNING
   - メモリ使用率 > 95%が1分間継続 → CRITICAL

4. **ビジネス指標**:
   - 検索精度 < 0.7が1時間継続 → WARNING
   - 検索可能文書率 < 90%が30分継続 → WARNING



### 2. 通知サービス

**注**: 専用の通知サービスモジュール（`app/services/notification_service.py`）は未実装ですが、アラートサービスに統合されています。

#### 通知システムの設計

アラートの重要度と緊急性に応じて、複数の通知チャネルを使い分けます。

**通知チャネル**:

1. **Slack通知**:
   - 全てのアラートを指定チャンネルに通知
   - 色分けによる視認性を重視
   - INFO: 緑色、WARNING: オレンジ、CRITICAL: 赤色

2. **Microsoft Teams通知**:
   - エンタープライズ環境向け
   - Adaptive Card形式でのリッチな表示
   - アクションボタンによる迅速な対応

3. **Email通知**:
   - CRITICALアラートのみ送信
   - 担当者・マネージャーに直接通知
   - 詳細なコンテキスト情報を含む

4. **PagerDuty連携**:
   - オンコールエンジニアへのエスカレーション
   - インシデント管理との統合
   - 自動ローテーション・エスカレーション

#### 通知ペイロードの設計

**共通情報**:

- アラートタイトルと説明
- 現在値と闾値
- 継続時間
- タイムスタンプ
- 影響範囲（サービス名、エンドポイント等）

**Slack固有情報**:

- Attachment形式での構造化表示
- 色分けによる重要度表示
- グラフやダッシュボードへのリンク

**Email固有情報**:

- HTML形式での読みやすいレイアウト
- 対応手順へのリンク
- 関連ログの添付

#### 通知の抑制・集約

**アラート疲れ防止のための機能**:

1. **クールダウン期間**:
   - 同じアラートの連続通知を抑制
   - デフォルト: 30分間

2. **集約機能**:
   - 5分間以内の同種アラートをまとめて通知
   - バッチ通知によるノイズ低減

3. **時間帯制限**:
   - 営業時間外のINFOアラートを抑制
   - CRITICALは24時間通知

4. **重複除去**:
   - 複数のメトリクスから同じ問題を検出した場合
   - 最も重要度の高いアラートのみ通知

#### 通知管理の設計哲学

**日本語解説**:
```
通知システムの設計原則：

1. 適切な通知先の選択（Right Channel Selection）
   - 緊急度に応じたチャネル選択
   - 受信者の勤務時間考慮
   - エスカレーションパスの明確化

2. 情報の構造化（Structured Information）
   - 一目で状況が分かる表示
   - 必要な情報の網羅
   - アクション可能な内容

3. ノイズ削減（Noise Reduction）
   - インテリジェントな集約
   - 重複通知の排除
   - コンテキストに基づくフィルタリング

4. 確実な到達性（Reliable Delivery）
   - 複数チャネルでの冗長性
   - 配信確認とリトライ
   - フォールバック機構

通知抑制のベストプラクティス：
- 同一アラートは30分のクールダウン
- 5分以内の類似アラートはバッチ化
- 深夜のINFOレベル通知は抑制
- ビジネスアワー外は管理者のみ通知
```

---

## ❗ よくある落とし穴と対策

### 1. ログの機密情報漏洩

```python
# ❌ 問題: パスワードやトークンをログ出力
logger.info(f"User login: {request_data}")  # パスワード含む

# ✅ 対策: 機密データフィルタリング
def sanitize_log_data(data: dict) -> dict:
    """ログデータの機密情報除去"""
    sensitive_keys = ['password', 'token', 'api_key', 'secret']

    sanitized = data.copy()
    for key in sensitive_keys:
        if key in sanitized:
            sanitized[key] = '***REDACTED***'

    return sanitized

logger.info(f"User login: {sanitize_log_data(request_data)}")
```

### 2. 無制限ログ蓄積

```python
# ❌ 問題: ログローテーション設定なし
# ログファイルが無制限に増大

# ✅ 対策: ログローテーション設定
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'app.log',
    maxBytes=100*1024*1024,  # 100MB
    backupCount=10           # 10ファイル保持
)
```

### 3. アラート疲れ

```python
# ❌ 問題: 過敏なアラート設定
AlertRule(
    threshold=0.01,  # 1%エラー率で即アラート
    duration=10      # 10秒継続で通知
)

# ✅ 対策: 適切な閾値・継続時間設定
AlertRule(
    threshold=0.05,  # 5%エラー率
    duration=300,    # 5分継続
    cooldown=1800    # 30分のクールダウン
)
```

### 4. エラーコンテキストの欠如

```python
# ❌ 問題: エラー時の情報不足
try:
    result = process_data(data)
except Exception as e:
    logger.error(f"Processing failed: {e}")

# ✅ 対策: 豊富なコンテキスト情報
try:
    result = process_data(data)
except Exception as e:
    logger.error(
        "Data processing failed",
        exc_info=True,  # スタックトレース含む
        extra={
            "data_id": data.id,
            "data_size": len(data),
            "user_id": current_user.id,
            "operation": "process_data",
            "error_type": type(e).__name__
        }
    )
```

---

## 🎯 理解確認のための設問

### エラーハンドリング理解

1. カスタム例外でエラーカテゴリとエラー重要度を分ける理由を説明してください
2. リトライ処理で指数バックオフとジッターを使用する目的を説明してください
3. サーキットブレーカーパターンが有効な場面と状態遷移を説明してください

### ログ管理理解

1. 構造化ログ（JSON形式）の利点と、必須で含めるべき情報を5つ挙げてください
2. セキュリティ監査ログで記録すべきイベントを6つ挙げてください
3. ログレベル（DEBUG/INFO/WARN/ERROR/CRITICAL）の使い分け基準を説明してください

### メトリクス理解

1. Counter、Gauge、Histogramメトリクスの違いと適用場面を説明してください
2. RAGシステム特有のメトリクス（検索精度・再現率）の計算方法を説明してください
3. アラートルールで継続時間（duration）を設定する理由と適切な値の決め方を説明してください

### 監視・運用理解

1. 効果的なアラート設計で避けるべき3つの問題とその対策を説明してください
2. 障害対応時に確認すべきログ・メトリクスの優先順位を説明してください
3. システムの可観測性（Observability）を向上させるための3つの要素を説明してください

---

## 📚 次のステップ

エラーハンドリングと監視を理解できたら、最終段階に進んでください：

- **Step08**: デプロイメントと運用 - Docker・Kubernetes・CI/CD・本番運用

適切なエラーハンドリングと監視は、システムの信頼性と運用効率を大幅に向上させます。次のステップでは、これらの実装をコンテナ環境にデプロイし、本番運用するための手法を学習します。
