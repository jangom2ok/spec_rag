# Embedding

## 選定方針

システム開発における仕様書や用語情報などをまとめたRAGシステムを構築したい。
常に更新されるシステム開発に関する情報を扱う性質上、仕様追加や、用語追加が頻繁に発生する。
また、このRAGは外部システムから参照される前提で構築する。
その上で最適なEmbeddingはなにか？

## 選定方針 — 「何を最適化したいか」を言語化する

RAG 向け Embedding を選ぶときは、まず **評価軸** を整理します。

| 軸            | なぜ重要か                       | 典型的な要件                                 |
| ------------ | --------------------------- | -------------------------------------- |
| **多言語性能**    | 仕様書やソース注釈は日英混在が多い           | 日本語・英語どちらのクエリでも同品質で検索できるか              |
| **ドメイン適合**   | IT 技術用語・コード断片を理解できるか        | “K8s Ingress” や `func main()` 等を意味で捉える |
| **入力量**      | 1 行用語〜 100 p 超え設計書まで長さがばらつく | 8 k token 以上を 1 shot で埋め込めるか           |
| **更新コスト**    | 毎日ドキュメント追加→即検索に反映したい        | 1 秒あたり千件以上の埋め込み速度／API 料金               |
| **運用・ライセンス** | 外部 API かセルフホストか             | 機密データ・SLA・予算に合致するか                     |

---

## 候補モデルと特徴

| 推奨度\* | モデル                                        | 主要特徴                                                                                                                        | ライセンス / 運用          | 参考                                                            |
| ----- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------- |
| ★★★   | **BAAI / BGE‑M3**                          | *Multi‑Lingual (100 + 言語)*、*Multi‑Granular (～8 192 tok)*、Dense + Sparse + Multi‑Vector の **ハイブリッド** を 1 モデルで生成。日本語技術文書で高精度。 | Apache‑2.0（セルフホスト可） | ([huggingface.co][1], [bge-model.com][2], [deepinfra.com][3]) |
| ★★☆   | **OpenAI text‑embedding‑3‑large**          | 商用 API としては現状最高クラスの MTEB スコア (64.6)。圧縮 256 次元でも高精度。                                                                         | 外部 API・従量課金         | ([clarifai.com][4], [openai.com][5])                          |
| ★★☆   | **Cohere embed‑multilingual‑v3**           | 1024 dim、100 + 言語対応。日本語でも OpenAI に迫る精度。オンプレ版（OCI, AWS Marketplace）あり。                                                       | 商用 API／BYO モデル      | ([cohere.com][6], [docs.pinecone.io][7], [clarifai.com][8])   |
| ★★    | **intfloat / E5‑large‑v2（multilingual）**   | 軽量・高速。Open‑source で GPU1 枚でも 5 k doc/s 以上埋め込み可。                                                                             | Apache‑2.0          | ([beam.cloud][9], [elastic.co][10])                           |
| ★★    | **専門特化：Qwen3‑Embedding / Code‑Embeddings** | コード検索や API シグネチャ検索が多い場合に併用。                                                                                                 | Apache‑2.0 ほか       | ([github.com][11], [docs.mistral.ai][12])                     |

*\* 推奨度は本ユースケース（日本語 × 技術ドメイン × 頻繁更新）の適合度を 3 段階で示しています。*

---

## なぜ BGE‑M3 を第一候補に推すか

1. **1 モデルで 3 役**

   * Dense（意味ベクトル）
   * Sparse（BM25 代替の語彙重み）
   * Multi‑Vector（長文内の局所的特徴を保持）
     → 技術用語の「表記ゆれ」や 100 ページ級仕様書の章内ヒット率が向上。([huggingface.co][1])

2. **長文対応 (8 k tok)**
   システム要件定義書や Swagger YAML など “1 ファイル＝超長文” を分割せずに格納可能。([bge-model.com][2])

3. **完全オンプレ運用が容易**
   Apache‑2.0 なので機密仕様も外に出さず運用可。GPU‑RAM 24 GB で ≈1 k doc/s の埋め込みが現実的。

4. **ハイブリッド検索に追加コスト不要**
   Dense ベクトル生成時に Sparse 重みを同時取得できるため、RRF\* や Vespa/Milvus の Hybrid Search をすぐ実装可能。([huggingface.co][1])
   \*RRF: Reciprocal Rank Fusion

---

## 運用アーキテクチャ例

```plaintext
開発者                         外部システム
   │                              │
   │  (REST / gRPC)               │
   ▼                              ▼
┌────────────────┐     ┌──────────────────┐
│   RAG API      │<--─▶│ Vector DB        │
│ (FastAPI etc.) │     │ (Milvus, Qdrant) │
└────────────────┘     └──────────────────┘
         ▲                           ▲
         │                           │
         │   Streaming ETL           │
         │  (毎朝/即時)               │
         ▼                           │
    ┌───────────────────────┐        │
    │ Doc Collector         │────────┘
    │ (Git, Confluence,     │
    │   Swagger, Sheets …)  │
    └───────────────────────┘
```

* **Collector** が変更差分だけを抽出 → BGE‑M3 で埋め込み → Vector DB に *upsert*
* 外部システムは **RAG API** を叩くだけで常に最新仕様書を検索・参照可能。

### 埋め込み更新ポリシー

| 変化         | 処理                   | 目安     |
| ---------- | -------------------- | ------ |
| 文書追加       | **部分埋め込み & upsert**  | 秒〜分    |
| 大規模リファクタ   | **全文再埋め込み**          | 週次     |
| モデルアップグレード | **並列 Index** → 検証後切替 | 四半期〜半年 |

---

## “さらに精度を上げたい” ときの 3 ステップ

1. **自社データで微調整 (Instruction‑tuning)**
   Databricks の事例ではタスク特化微調整で 5 〜 15 pt 向上。([databricks.com][13])
2. **再ランク用 Reranker**
   BGE‑M3 系の `bge-reranker-v2-m3` を top‑k に適用すると精度が大幅に上がる。([aimodels.fyi][14])
3. **コード専用サブインデックス**
   Qwen3‑Embedding などコード特化モデルでソースファイルだけ別ベクトル化し、二段検索。

---

## 結論

* **ベースライン**：BGE‑M3 をセルフホストし、Dense+Sparse ハイブリッドで即時検索を実現。
* **性能上限を追う場合**：OpenAI text‑embedding‑3‑large or Cohere v3 を比較 A/B。
* **技術ドメイン適合**：自社 Issue/PR/設計書で 追加微調整 or コード用 Embedding を併用。

この構成で、**「毎日変わる開発仕様を途切れなく外部提供できる RAG」** を、コストと精度の両面で最適化できます。

[1]: https://huggingface.co/BAAI/bge-m3?utm_source=chatgpt.com "BAAI/bge-m3 - Hugging Face"
[2]: https://bge-model.com/bge/bge_m3.html?utm_source=chatgpt.com "BGE-M3 — BGE documentation - BGE Models"
[3]: https://deepinfra.com/BAAI/bge-m3-multi?utm_source=chatgpt.com "BAAI/bge-m3-multi - Demo - DeepInfra"
[4]: https://clarifai.com/openai/embed/models/text-embedding-3-large?utm_source=chatgpt.com "text-embedding-3-large model | Clarifai - The World's AI"
[5]: https://openai.com/index/new-embedding-models-and-api-updates/?utm_source=chatgpt.com "New embedding models and API updates - OpenAI"
[6]: https://cohere.com/blog/introducing-embed-v3?utm_source=chatgpt.com "Introducing Embed v3 - Cohere"
[7]: https://docs.pinecone.io/models/cohere-embed-multilingual-v3.0?utm_source=chatgpt.com "embed-multilingual-v3.0 | Cohere - Pinecone Docs"
[8]: https://clarifai.com/cohere/embed/models/cohere-embed-multilingual-v3_0?utm_source=chatgpt.com "cohere-embed-multilingual-v3_0 model | Clarifai - The World's AI"
[9]: https://www.beam.cloud/blog/best-embedding-models?utm_source=chatgpt.com "Choosing the Best Embedding Models for RAG and Document ..."
[10]: https://www.elastic.co/search-labs/blog/multilingual-vector-search-e5-embedding-model?utm_source=chatgpt.com "Multilingual vector search with the E5 embedding model - Elastic"
[11]: https://github.com/QwenLM/Qwen3-Embedding?utm_source=chatgpt.com "QwenLM/Qwen3-Embedding - GitHub"
[12]: https://docs.mistral.ai/capabilities/embeddings/code_embeddings/?utm_source=chatgpt.com "Code Embeddings - Mistral AI Documentation"
[13]: https://www.databricks.com/blog/improving-retrieval-and-rag-embedding-model-finetuning?utm_source=chatgpt.com "Improving Retrieval and RAG with Embedding Model Finetuning"
[14]: https://www.aimodels.fyi/creators/huggingFace/BAAI?utm_source=chatgpt.com "Baai - Find Top AI Models on Hugging Face - AIModels.fyi"
