# Step05: データモデル設計とスキーマ詳細

## 🎯 この章の目標

PostgreSQL・ApertureDBでのデータモデル設計、スキーマ詳細、インデックス戦略、データ整合性の仕組みを理解する

---

## 📋 概要

RAGシステムでは、構造化データ（メタデータ）と非構造化データ（ベクター）を効率的に管理するため、
PostgreSQLとApertureDBを使い分けています。適切なスキーマ設計により、高速検索と拡張性を両立します。

### 🏗️ データベース構成

```text
データ保存戦略
├── PostgreSQL        # 構造化データ・メタデータ
│   ├── documents     # ドキュメント基本情報
│   ├── chunks        # チャンク詳細
│   ├── sources       # ソース管理
│   └── users         # ユーザー・認証
├── ApertureDB        # ベクターデータ
│   ├── dense_descriptor_set    # Dense vectors
│   ├── sparse_descriptor_set   # Sparse vectors
│   └── multi_descriptor_set    # Multi-vectors
└── Redis             # キャッシュ・セッション
    ├── search_cache  # 検索結果キャッシュ
    ├── embedding_cache # 埋め込みキャッシュ
    └── session_store # ユーザーセッション
```

---

## 🗃️ PostgreSQL スキーマ設計

### 1. ドキュメント管理テーブル

#### `documents` テーブル

**実装ファイル**: `../../app/models/schema/documents.sql`

ドキュメント管理の中核となるテーブルで、すべての文書情報を統括管理します。

**主要フィールドの説明**:

1. **基本情報**:
   - `id`: UUIDによる一意識別子
   - `title`, `content`: ドキュメントのタイトルと本文
   - `source_type`, `source_id`: ソースシステムの識別情報
   - `source_url`: 元文書へのリンク

2. **メタデータ**:
   - `author`: 作成者情報
   - `language`: 2文字の言語コード（ISO 639-1）
   - `category`: カテゴリ分類
   - `tags`: PostgreSQL配列で管理するタグ情報

3. **統計情報**:
   - `word_count`, `char_count`: 文書のサイズ情報
   - `chunk_count`: 分割されたチャンク数

4. **状態管理**:
   - `processing_status`: 処理状態（pending/processing/completed/failed）
   - `indexing_status`: インデックス作成状態
   - `error_message`: エラー時の詳細情報

**インデックス最適化**:

- ソース別検索用の複合インデックス
- ステータス監視用のインデックス
- タグ検索用のGINインデックス
- 日本語全文検索用のto_tsvectorインデックス

**制約の目的**:

- ステータス値の限定によるデータ整合性保証
- 言語コードの正規表現検証
- カウント値の非負数保証

#### `document_chunks` テーブル

**実装ファイル**: `../../app/models/schema/document_chunks.sql`

ドキュメントを検索可能なチャンクに分割して管理するテーブルです。

**主要フィールドの説明**:

1. **チャンク情報**:
   - `content`: チャンクの実際のテキスト内容
   - `chunk_type`: チャンクの種別（text/code/table等）
   - `position`: 親ドキュメント内での位置（順序保持）

2. **構造情報**:
   - `section_title`: 所属セクションのタイトル
   - `hierarchy_level`: 階層の深さ（1=トップレベル）
   - `parent_chunk_id`: 親チャンクへの参照（階層構造）

3. **ベクターマッピング**:
   - `dense_vector_id`: ApertureDB内のDense Vector ID
   - `sparse_vector_id`: ApertureDB内のSparse Vector ID
   - `multi_vector_id`: ApertureDB内のMulti-Vector ID
   - これらのIDを使用してPostgreSQLとApertureDBを連携

4. **性能最適化**:
   - `last_search_score`: 最後の検索スコアをキャッシュ
   - `search_count`: 検索頻度を追跡
   - 人気チャンクの分析に活用

**インデックス戦略**:

- ドキュメント単位のチャンク取得用複合インデックス
- 埋め込み状態でのフィルタリング
- ベクターIDによる高速ルックアップ
- 階層構造の効率的なトラバース

### 2. ソース管理テーブル

#### `sources` テーブル

**実装ファイル**: `../../app/models/schema/sources.sql`

外部データソースの接続情報と同期状態を管理するテーブルです。

**主要フィールドの説明**:

1. **ソース識別**:
   - `source_type`: ソースの種別（git/jira/confluence等）
   - `name`: ソースの表示名
   - 種別と名前の組み合わせで一意性保証

2. **接続設定**:
   - `connection_config`: JSONB形式で柔軟な設定保存
   - `credentials_encrypted`: 暗号化された認証情報
   - ソースタイプごとに異なる設定に対応

3. **同期管理**:
   - `sync_schedule`: Cron式で定期同期を設定
   - `last_sync_at`, `next_sync_at`: 同期タイミング管理
   - `sync_status`: 現在の同期状態

4. **統計とエラー管理**:
   - ドキュメント数の統計情報
   - エラーカウントと最終エラーメッセージ
   - 問題のあるソースの特定と対処

**インデックスの目的**:

- アクティブなソースの次回同期時刻でのソート
- ソースタイプ別の高速フィルタリング
- 部分インデックスによる効率化

### 3. 認証・ユーザー管理

#### `users` テーブル

**実装ファイル**: `../../app/models/schema/users.sql`

ユーザー情報と認証・認可情報を管理するテーブルです。

**主要フィールドの説明**:

1. **認証情報**:
   - `email`, `username`: 一意の識別子
   - `password_hash`: bcryptでハッシュ化されたパスワード
   - メールアドレスの正規表現検証

2. **権限管理**:
   - `role`: 事前定義されたロール（viewer/editor/admin/super_admin）
   - `permissions`: 追加権限の配列
   - RBAC（Role-Based Access Control）の実現

3. **セキュリティ機能**:
   - `failed_login_attempts`: ブルートフォース攻撃対策
   - `locked_until`: アカウントロック機能
   - `password_changed_at`: パスワード変更履歴

4. **プロファイル情報**:
   - `full_name`: 表示名
   - `avatar_url`: プロフィール画像
   - `timezone`: タイムゾーン設定

**セキュリティ設計**:

- パスワードはハッシュ化して保存
- ログイン失敗回数の追跡
- 一時的なアカウントロック
- メールアドレスの検証

#### `api_keys` テーブル

**実装ファイル**: `../../app/models/schema/api_keys.sql`

APIアクセス用の認証キーを管理するテーブルです。

**主要フィールドの説明**:

1. **APIキー情報**:
   - `key_hash`: SHA-256でハッシュ化されたAPIキー
   - `key_prefix`: 表示用のプレフィックス（例: "sk_live_"）
   - `name`, `description`: キーの識別情報

2. **アクセス制御**:
   - `permissions`: 許可された操作の配列
   - `rate_limit_per_minute`: 分あたりのリクエスト上限
   - 柔軟なアクセス制御の実現

3. **利用統計**:
   - `usage_count`: 累計使用回数
   - `last_used_at`: 最終使用日時
   - 使用状況の分析と監視

4. **ライフサイクル管理**:
   - `is_active`: キーの有効/無効化
   - `expires_at`: 有効期限設定
   - 定期的なキーローテーションに対応

**セキュリティ考慮**:

- APIキーはハッシュ化して保存
- ユーザー削除時にカスケード削除
- レート制限による乱用防止

---

## 🔍 ApertureDB ディスクリプタセット設計

### 1. Dense Vector Descriptor Set

**実装ファイル**: `../../app/models/aperturedb.py` (DenseVectorDescriptorSetクラス)

セマンティック検索用の密ベクトルを格納するディスクリプタセットです。

**スキーマ設計の要点**:

1. **主キーとベクトル**:
   - `id`: PostgreSQLのdocument_chunks IDと連携
   - `vector`: 1024次元のDense Vector（BGE-M3）
   - コサイン類似度での検索に最適化

2. **メタデータフィールド**:
   - `document_id`: 親ドキュメントへの参照
   - `source_type`: フィルタリング用
   - `language`: 言語別検索に対応
   - `chunk_position`: 順序保持

3. **インデックス最適化**:
   - **HNSWアルゴリズム**: 高速な近似最近傍探索
   - **M=16**: メモリと精度のバランス
   - **efConstruction=256**: 高品質なインデックス構築

**パフォーマンス特性**:

- 検索速度: 100万ベクトルで約10ms
- インデックス構築: 10万ベクトルで約5分
- メモリ使用量: ベクトル数×4KB + インデックスオーバーヘッド

### 2. Sparse Vector Descriptor Set

**実装ファイル**: `../../app/models/aperturedb.py` (SparseVectorDescriptorSetクラス)

キーワード検索用の疎ベクトルを格納するディスクリプタセットです。

**Sparse Vectorの特徴**:

1. **ベクトル形式**:
   - 語彙IDと重みのマッピング
   - TF-IDFに似た重み付け
   - 解釈可能な語彙重要度

2. **追加メタデータ**:
   - `vocabulary_size`: 有効語彙数の記録
   - スパーシティ（疎度）の追跡
   - メモリ使用量の推定に活用

3. **インデックス最適化**:
   - **SPARSE_INVERTED_INDEX**: 転置インデックス
   - **drop_ratio_build=0.2**: 低頻度語の20%を除外
   - メモリ効率と検索精度のバランス

**ユースケース**:

- 特定の技術用語検索
- エラーメッセージ検索
- 完全一致が必要な場面

### 3. Multi-Vector Descriptor Set

**実装ファイル**: `../../app/models/aperturedb.py` (MultiVectorDescriptorSetクラス)

ColBERT式のトークンレベルベクトルを格納するディスクリプタセットです。

**Multi-Vectorの特徴**:

1. **トークンレベルの表現**:
   - 各トークンが独立した1024次元ベクトル
   - 文章内の局所的な特徴を保持
   - 細かい粒度でのマッチングが可能

2. **位置情報の保持**:
   - `token_count`: トークン数の記録
   - `token_positions`: JSON形式で位置情報を保存
   - 文脈を考慮した検索が可能

3. **使用シーン**:
   - 質問の一部と文書の一部の精密マッチング
   - 長文ドキュメント内の特定箇所検索
   - 複雑なクエリへの対応

**パフォーマンス考慮**:

- ストレージ要件が大きい
- 検索時の計算コストが高い
- 高精度が必要な場面で使用

---

## 🔗 データ関連付けとマッピング

### PostgreSQL ↔ ApertureDB 連携

**実装ファイル**: `../../app/services/vector_mapping_service.py`

VectorMappingServiceは、PostgreSQLの構造化データとApertureDBのベクターデータを連携させる
重要なサービスです。

**データ連携の設計思想**:

1. **トランザクション整合性**:
   - PostgreSQLへのメタデータ保存を先に実行
   - ApertureDBへのベクター保存が成功した場合のみ、PostgreSQLを更新
   - 失敗時は適切なロールバック処理を実行
   - 分散システムでのデータ整合性を保証

2. **IDマッピング戦略**:
   - PostgreSQLのチャンクIDをベースに、ベクタータイプごとのIDを生成
   - `{chunk_id}_dense`、`{chunk_id}_sparse`、`{chunk_id}_multi`の形式
   - これにより、ベクターからメタデータへの逆引きが可能

3. **保存プロセスの流れ**:
   - **ステップ1**: PostgreSQLにチャンクメタデータを保存（status: 'processing'）
   - **ステップ2**: 3種類のベクターをそれぞれのApertureDBディスクリプタセットに保存
   - **ステップ3**: 成功後、PostgreSQLのベクターIDを更新（status: 'completed'）

**エラーハンドリング**:

- ベクター保存失敗時は、PostgreSQLのレコードをクリーンアップ
- 詳細なエラーログを記録し、デバッグを容易に
- リトライ可能なエラーは自動的に再試行

**パフォーマンス最適化**:

- バッチ処理による効率的な保存
- 非同期処理による並列化
- コネクションプールの活用

---

## 📊 データ整合性と制約

### 1. 外部キー制約と削除カスケード

**実装ファイル**: `../../app/models/schema/triggers.sql`

ドキュメント削除時にApertureDBのベクターも適切に削除するため、
PostgreSQLトリガーを実装しています。

**削除カスケードの設計**:

1. **トリガーベースの削除**:
   - ドキュメント削除時に自動的に発火
   - 関連するすべてのチャンクのベクターIDを収集
   - 外部関数経由でApertureDBのベクターを削除

2. **非同期処理**:
   - ApertureDBへの削除リクエストは非同期で実行
   - PostgreSQLのトランザクションをブロックしない
   - 削除失敗時はログに記録し、バックグラウンドで再試行

3. **整合性保証**:
   - PostgreSQLの外部キー制約でチャンクの削除は保証
   - ApertureDBの削除は最終的整合性で処理
   - 定期的なクリーンアップジョブで不整合を解消

### 2. データ状態一貫性チェック

**実装ファイル**: `../../app/services/data_consistency_checker.py`

分散システムにおけるデータ整合性を定期的に検証し、
問題を早期発見するサービスです。

**整合性チェックの種類**:

1. **孤立チャンクの検出**:
   - 親ドキュメントが削除されたが、チャンクが残存しているケース
   - LEFT JOINを使用して効率的に検出
   - 検出されたチャンクは削除候補としてマーク

2. **欠損ベクターの検出**:
   - embedding_statusが'completed'だが、ベクターIDがNULLのケース
   - 再埋め込み処理の対象として特定
   - エラーログと照合して原因を分析

3. **ステータス不整合の検出**:
   - processing_statusとindexing_statusの矛盾
   - 長時間'processing'状態のドキュメント
   - タイムアウトしたタスクの検出

4. **インデックス整合性の確認**:
   - ApertureDBのインデックス構築状態
   - PostgreSQLのインデックス破損チェック
   - パフォーマンス劣化の原因特定

**自動修復機能**:

- 軽微な不整合は自動的に修復
- 重大な問題はアラートを発報
- 修復履歴をログに記録

---

## ⚡ パフォーマンス最適化

### 1. パーティショニング戦略

**実装ファイル**: `../../app/models/schema/partitioning.sql`

大量データを効率的に管理するため、
日付ベースのパーティショニングを実装しています。

**パーティショニングの設計思想**:

1. **月単位の分割**:
   - 各月のデータを独立したテーブルに格納
   - 古いデータの削除やアーカイブが容易
   - パーティションプルーニングによる検索高速化

2. **自動パーティション作成**:
   - 月次バッチで新しいパーティションを自動作成
   - パーティション不足によるエラーを防止
   - 運用負荷の軽減

3. **パフォーマンス効果**:
   - 日付範囲検索が大幅に高速化（最大10倍）
   - インデックス再構築時間の短縮
   - バキューム処理の効率化

**運用上の考慮事項**:

- パーティション数は最大24ヶ月分を保持
- 古いパーティションは自動的にアーカイブ
- パーティション統計情報の定期更新

### 2. インデックス最適化

**実装ファイル**: `../../app/models/schema/indexes.sql`

クエリパターンに基づいた最適なインデックス設計により、
検索性能を大幅に向上させています。

**インデックス設計の原則**:

1. **複合インデックスの活用**:
   - よく使用される検索条件の組み合わせを分析
   - カラムの順序は選択性の高い順に配置
   - カバリングインデックスによるインデックスオンリースキャン

2. **部分インデックスの効果**:
   - WHERE句の条件を満たすレコードのみインデックス化
   - インデックスサイズの削減（最大80%削減）
   - メンテナンスコストの低減

3. **並行インデックス作成**:
   - CONCURRENTLYオプションで本番環境でも安全に作成
   - テーブルロックを回避
   - 作成時間は長くなるが、サービス影響なし

**GINインデックスの活用**:

- 日本語全文検索に最適化
- to_tsvectorによる形態素解析結果をインデックス化
- LIKE検索より100倍以上高速

### 3. ApertureDB パフォーマンス設定

**実装ファイル**: `../../app/config/aperturedb_config.py`

ApertureDBベクターデータベースのパフォーマンスを最大化するための設定です。

**ディスクリプタセット設定の最適化**:

1. **シャード数の決定**:
   - データ量とクエリ負荷に基づいて設定
   - 2シャードで並列処理による高速化
   - 将来の拡張性を考慮した設計

2. **整合性レベルの選択**:
   - **Strong**: 完全な整合性保証（デフォルト）
   - **Bounded**: 有界整合性（パフォーマンス優先時）
   - **Eventually**: 最終的整合性（最高速度）

**HNSWインデックスパラメータ**:

1. **M（接続数）**:
   - 16: メモリ使用量と検索精度のバランス点
   - 大きいほど精度向上、メモリ消費増
   - 本番環境での推奨値: 16-32

2. **efConstruction（構築品質）**:
   - 256: 高品質なインデックス構築
   - 構築時間は増加するが、検索性能向上
   - 一度構築すれば長期間使用可能

3. **ef（検索品質）**:
   - 128: リアルタイム性と精度のバランス
   - 動的に調整可能
   - クエリごとに変更可能

**メモリ管理**:

- インデックス構築時のメモリ上限設定
- OOMエラーの防止
- 複数コレクション間でのリソース配分

---

## ❗ よくある落とし穴と対策

### 1. ベクター次元不一致

**対策実装**: `../../app/services/vector_validator.py`

**問題の背景**:

- 異なるモデルバージョンでの次元数変更
- 手動でのベクター操作によるミス
- 検索時の予期しないエラー

**ベストプラクティス**:

1. **事前検証の徹底**:
   - ベクター保存前に必ず次元数を確認
   - BGE-M3の期待次元数（1024）との一致を検証
   - 不一致時は明確なエラーメッセージ

2. **正規化処理**:
   - L2正規化による単位ベクトル化
   - コサイン類似度計算の安定化
   - 数値精度の向上

3. **モデルバージョン管理**:
   - 使用モデルのバージョンを記録
   - バージョン変更時の移行計画
   - 後方互換性の維持

### 2. トランザクション境界の問題

**対策実装**: `../../app/services/saga_transaction_manager.py`

**分散トランザクションの課題**:

- PostgreSQLとApertureDBは独立したシステム
- 従来のACIDトランザクションが使用不可
- 部分的な失敗による不整合リスク

**Sagaパターンの実装**:

1. **補償トランザクション**:
   - 各操作に対応する取り消し処理を定義
   - 失敗時は逆順で補償処理を実行
   - 最終的整合性を保証

2. **実装のポイント**:
   - 冪等性の確保（同じ操作を複数回実行しても安全）
   - タイムアウト設定による無限待機の防止
   - 詳細なログによる問題追跡

3. **エラーハンドリング**:
   - 補償処理自体の失敗も考慮
   - 手動介入が必要なケースの検出
   - アラート通知による迅速な対応

**運用上の考慮**:

- 定期的な整合性チェック
- 不整合データの自動修復
- 監視ダッシュボードでの可視化

### 3. メモリリークとコネクション管理

**実装ファイル**: `../../app/core/database_manager.py`

**リソース管理の重要性**:

- 長時間稼働でのメモリリーク防止
- コネクション枯渇の回避
- 適切なリソース解放の保証

**コンテキストマネージャーパターン**:

1. **自動リソース管理**:
   - `async with`構文での安全な利用
   - 例外発生時も確実にクリーンアップ
   - リソースリークの完全防止

2. **コネクションプール設定**:
   - **min_size**: 最小接続数（アイドル時）
   - **max_size**: 最大接続数（ピーク時）
   - **command_timeout**: クエリタイムアウト

3. **プール最適化**:
   - PostgreSQL: 5-20接続（CPU数×2-4）
   - ApertureDB: 10接続（並行検索数に依存）
   - Redis: 20接続（キャッシュアクセス頻度）

**監視項目**:

- アクティブ接続数
- 待機中のクエリ数
- 接続エラー率
- メモリ使用量の推移

**トラブルシューティング**:

- 接続リークの検出方法
- デッドロックの回避
- タイムアウト値の調整

---

## 🎯 理解確認のための設問

### スキーマ設計理解

1. `documents`テーブルで`processing_status`と`indexing_status`を

   分けている理由を説明してください

2. `document_chunks`テーブルの`parent_chunk_id`フィールドの用途と

   階層構造の表現方法を説明してください

3. ApertureDBでDense、Sparse、Multi-Vectorを別ディスクリプタセットに分ける

   メリットを3つ挙げてください

### データ整合性理解

1. PostgreSQL-ApertureDB間のデータ整合性を保つために

   実装された仕組みを説明してください

2. 分散トランザクションでSagaパターンを使用する理由と

   補償処理の重要性を説明してください

3. `cleanup_document_vectors()`トリガー関数が必要な理由を説明してください

### パフォーマンス理解

1. 大量データ処理でパーティショニングが有効な理由と、

   適切な分割戦略を説明してください

2. ApertureDBインデックスパラメータ（M、efConstruction）の

   調整指針を説明してください

3. 部分インデックスを使用することの利点と適用場面を説明してください

### 運用理解

1. データ整合性チェッカーで検出すべき4種類の不整合と

   その影響を説明してください

2. ベクター次元不一致が発生する原因と事前防止策を説明してください
3. コネクションプール設定で考慮すべきパラメータを5つ挙げてください

---

## 📚 次のステップ

データモデル設計を理解できたら、次の学習段階に進んでください：

- - **Step06**: 認証・認可システム - JWT・API Key認証の実装詳細
- **Step07**: エラーハンドリングと監視 - 例外処理・ログ・メトリクス収集
- **Step08**: デプロイメントと運用 - Docker・Kubernetes・CI/CD

適切なデータモデル設計は、システムの拡張性と保守性を決定する重要な要素です。
次のステップでは、このデータを安全に保護する認証・認可システムについて学習します。
