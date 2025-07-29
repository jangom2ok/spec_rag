# RAGシステム データ投入ガイド

## 概要

このドキュメントでは、RAGシステムにデータを投入する方法について説明します。システムは複数のデータソースからの投入をサポートしており、REST API経由での投入が基本となります。

## データ投入の流れ

### 1. ファイル読み込み時

- ファイルの全内容がcontentフィールドに読み込まれます（line 128）
- ファイル名、ファイルパス、ファイルタイプなどのメタデータも保存されます

### 2. 処理フロー

- ファイル読み込み: ファイルの全内容を文字列として読み込み
- チャンク分割: 大きなファイルは適切なサイズのチャンクに分割
- ベクトル化: 各チャンクの内容をBGE-M3でベクトル化
- 保存:
  - PostgreSQL: メタデータとテキスト内容
  - ApertureDB: ベクトルデータ

### 3. API経由での投入時も同様

```json
{
  "title": "ドキュメントタイトル",
  "content": "ここにファイルの全内容が入ります...",  // ← ファイルの中身
  "source_type": "test"
}
```

### 4. 検索時の動作

- ファイルの内容に基づいてベクトル検索が行われる
- キーワード検索も内容に対して実行される
- 検索結果には該当する内容の一部が返される

つまり、ファイルの中身は完全に読み込まれ、検索可能な形で保存されます。

## 対応データソース

### 現在サポートされているデータソース

1. **ファイルシステム** (`file`)
   - ローカルファイルの直接読み込み
   - 対応形式: `*.txt`, `*.md`など
   - 実装状態: ✅ 本番利用可能

2. **テストソース** (`test`)
   - 開発・検証用のモックデータ
   - 実装状態: ✅ 開発環境で利用可能

### 実装中のデータソース（開発中）

以下のデータソースは現在開発中で、本番環境では利用できません：

1. **Confluence** (`confluence`)
   - Atlassian Confluenceからの文書取得
   - APIトークン認証が必要
   - 実装状態: ⚠️ 開発中

2. **JIRA** (`jira`)
   - JIRAチケットからの情報取得
   - APIトークン認証が必要
   - 実装状態: ⚠️ 開発中

## データ投入方法

### 1. 単一ドキュメントの投入

個別のドキュメントをAPI経由で投入する方法です。

#### 単一ドキュメントの投入エンドポイント

```http
POST /v1/documents/
```

#### 単一ドキュメントの投入リクエスト例

```bash
curl -X POST http://localhost:8000/v1/documents/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "title": "API設計ガイド",
    "content": "FastAPIを使用したREST API設計のベストプラクティス...",
    "source_type": "test"
  }'
```

注: 現在、`source_type`として利用可能なのは`"test"`のみです。

#### レスポンス例

```json
{
  "id": "doc_123456",
  "title": "API設計ガイド",
  "content": "FastAPIを使用したREST API設計のベストプラクティス...",
  "source_type": "test"
}
```

### 2. バッチ処理による投入

複数のドキュメントを一括で処理する方法です。

#### バッチ処理エンドポイント

```http
POST /v1/documents/process
```

#### バッチ処理リクエスト例

```bash
curl -X POST http://localhost:8000/v1/documents/process \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "source_type": "file",
    "source_path": "/opt/documents",
    "file_patterns": ["*.md", "*.txt"],
    "batch_size": 50,
    "max_documents": 1000,
    "chunking_strategy": "semantic",
    "chunk_size": 1000,
    "overlap_size": 200,
    "extract_structure": true,
    "extract_entities": true,
    "extract_keywords": true,
    "max_concurrent_documents": 5
  }'
```

#### パラメータ説明

| パラメータ | 型 | 説明 | デフォルト値 |
|-----------|-----|------|-------------|
| `source_type` | string | データソースタイプ | 必須 |
| `source_path` | string | ファイルパス（fileタイプの場合） | null |
| `file_patterns` | array | 読み込むファイルパターン | ["*.txt", "*.md"] |
| `batch_size` | integer | バッチサイズ | 10 |
| `max_documents` | integer | 最大処理ドキュメント数 | 100 |
| `chunking_strategy` | string | チャンク分割戦略 | "fixed_size" |
| `chunk_size` | integer | チャンクサイズ | 1000 |
| `overlap_size` | integer | チャンク間のオーバーラップ | 200 |

### 3. 同期処理による投入

処理完了を待機する同期的な投入方法です。

#### エンドポイント

```http
POST /v1/documents/process/sync
```

使用方法はバッチ処理と同じですが、処理が完了するまでレスポンスを待機します。

## 認証方法

### API Key認証

```bash
-H "X-API-Key: your-api-key"
```

### JWT認証

```bash
-H "Authorization: Bearer your-jwt-token"
```

## 外部システムからの投入

### Googleドライブからのデータ投入

現在、Googleドライブからの直接投入はサポートされていません。以下の方法で実現可能です：

#### 方法1: 外部ツール経由（推奨）

```python
# 外部ツールの実装例
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build

class GoogleDriveToRAG:
    def __init__(self, rag_api_key, google_credentials_path):
        self.rag_api_key = rag_api_key
        self.rag_base_url = "http://localhost:8000"

        # Google Drive API初期化
        credentials = service_account.Credentials.from_service_account_file(
            google_credentials_path,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        self.drive_service = build('drive', 'v3', credentials=credentials)

    def fetch_and_upload_document(self, file_id):
        # Google Driveからファイルを取得
        file = self.drive_service.files().get(fileId=file_id).execute()
        content = self.drive_service.files().get_media(fileId=file_id).execute()

        # RAGシステムに投入
        response = requests.post(
            f"{self.rag_base_url}/v1/documents/",
            headers={
                "X-API-Key": self.rag_api_key,
                "Content-Type": "application/json"
            },
            json={
                "title": file['name'],
                "content": content.decode('utf-8'),
                "source_type": "test"  # 現状はtestを使用
            }
        )
        return response.json()
```

### Slackからのデータ投入

Slackからのデータ投入も外部ツール経由で実現できます：

```python
# 外部ツールの実装例
import requests
from slack_sdk import WebClient

class SlackToRAG:
    def __init__(self, rag_api_key, slack_token):
        self.rag_api_key = rag_api_key
        self.rag_base_url = "http://localhost:8000"
        self.slack_client = WebClient(token=slack_token)

    def fetch_and_upload_messages(self, channel_id, limit=100):
        # Slackからメッセージを取得
        result = self.slack_client.conversations_history(
            channel=channel_id,
            limit=limit
        )

        documents = []
        for message in result['messages']:
            # RAGシステムに投入
            response = requests.post(
                f"{self.rag_base_url}/v1/documents/",
                headers={
                    "X-API-Key": self.rag_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "title": f"Slack Message - {message['ts']}",
                    "content": message['text'],
                    "source_type": "test"
                }
            )
            documents.append(response.json())

        return documents
```

## 処理状況の確認

### 個別ドキュメントの処理状況

```bash
GET /v1/documents/process/status/{document_id}
```

### 全ドキュメントの処理状況

```bash
GET /v1/documents/process/status
```

## ベストプラクティス

1. **バッチサイズの調整**
   - 大量のドキュメントを処理する場合は、適切なバッチサイズ（50-100）を設定
   - メモリ使用量とパフォーマンスのバランスを考慮

2. **チャンク戦略の選択**
   - `fixed_size`: 固定サイズでの分割（技術文書向け）
   - `semantic`: 意味単位での分割（自然言語文書向け）
   - `hierarchical`: 階層構造を考慮した分割（構造化文書向け）

3. **エラーハンドリング**
   - 処理状況を定期的に確認
   - 失敗したドキュメントは個別に再処理

4. **認証情報の管理**
   - API Keyは環境変数で管理
   - 定期的なキーのローテーション

5. **現在利用可能なソースタイプ**
   - 本番環境では`file`と`test`のみが利用可能
   - ConfluenceやJIRA統合は開発中のため、外部ツール経由での投入を推奨

## トラブルシューティング

### よくあるエラー

1. **認証エラー (401)**

   ```json
   {
     "error": {
       "code": "AUTHENTICATION_ERROR",
       "message": "Authentication required"
     }
   }
   ```

   **対処法**: API KeyまたはJWTトークンを確認

2. **権限エラー (403)**

   ```json
   {
     "error": {
       "code": "AUTHORIZATION_ERROR",
       "message": "Write permission required"
     }
   }
   ```

   **対処法**: ユーザーの権限設定を確認

3. **処理エラー (500)**

   ```json
   {
     "error": {
       "code": "PROCESSING_ERROR",
       "message": "Document processing failed"
     }
   }
   ```

   **対処法**: ドキュメントの形式やサイズを確認

## 今後の拡張予定

- JIRA統合の本番実装
- Google Drive統合モジュール
- Slack統合モジュール
- GitHub統合
- リアルタイムストリーミング投入

## 関連ドキュメント

- [API仕様書](../learning/Step02_API_Layer.md)
- [データフロー](../learning/Step01_DataFlow.md)
- [認証ガイド](../learning/Step06_Authentication.md)
