# 開発サイクル（TDDベース）

本ドキュメントは、RAGシステム開発におけるTDD（テスト駆動開発）を基本とした開発サイクルと、各ステップで利用する具体的なコマンド例をまとめたものです。

---

## 1. テストコードの作成

- 新機能や修正対象のテストを `tests/` 配下に作成

例：

```bash
vim tests/test_xxx.py
```

---

## 2. テストの実行（失敗を確認）

- まずテストが失敗することを確認

例：

```bash
pytest
```

---

## 3. 実装コードの追加・修正

- テストが通るように `app/` 配下のコードを実装

例：

```bash
vim app/xxx.py
```

---

## 4. テストの再実行（成功を確認）

- テストがパスすることを確認

例：

```bash
pytest
```

---

## 5. リファクタリング

- 必要に応じて実装やテストを整理

例：

```bash
vim app/xxx.py
vim tests/test_xxx.py
```

---

## 6. コード品質チェック

- flake8等で静的解析

例：

```bash
flake8 app/
```

---

## 7. コミット・プッシュ

- 変更をGitで管理

例：

```bash
git add .
git commit -m "Add: 新機能xxxのTDD実装"
git push origin feature/xxx
```

---

## 8. CI/CDによる自動テスト

- プルリクエスト作成時にGitHub Actions等で自動テスト

---

## 参考

テストカバレッジ計測：

```bash
pytest --cov=app
```

Docker Composeでの開発環境起動：

```bash
docker-compose up --build
```
