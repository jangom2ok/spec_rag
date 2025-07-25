# MCP

## Serenaのインストール

```fish
git clone https://github.com/oraios/serena
cd /path/work/git/serena
mkdir ~/.serena
cp src/serena/resources/serena_config.template.yml ~/.serena/serena_config.yml
```

## Serena MCP Server起動とプロジェクトのインデックス作成

```fish
cd /path/work/git/serena
uv run serena-mcp-server
```

別ターミナルを開き、プロジェクトのディレクトリでインデックスを作成

```fish
cd /path/work/git/spec_rag
uv run --directory /path/work/git/serena index-project
```

## ClaudeCodeへ追加

```fish
cd /path/work/git/spec_rag
# claude mcp add serena -- <serena-mcp-server> --context ide-assistant --project $(pwd)
claude mcp add serena -- uv run --directory /path/work/git/serena serena-mcp-server --context ide-assistant --project $(pwd)
```
