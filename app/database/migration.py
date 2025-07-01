"""データベースマイグレーション管理"""

from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine


class MigrationManager:
    """データベースマイグレーション管理クラス"""

    def __init__(self, database_url: str, migrations_dir: Path | None = None):
        self.database_url = database_url
        self.migrations_dir = migrations_dir or Path("migrations")
        self.config: Config | None = None

    def _get_config(self) -> Config:
        """Alembic設定を取得"""
        if self.config is None:
            self.config = create_alembic_config(
                database_url=self.database_url, migrations_dir=self.migrations_dir
            )
        return self.config

    async def initialize_migrations(self) -> dict[str, Any]:
        """マイグレーション環境を初期化"""
        config = self._get_config()

        # マイグレーションディレクトリが存在しない場合は作成
        if not self.migrations_dir.exists():
            self.migrations_dir.mkdir(parents=True)

            # Alembicの初期化
            command.init(config, str(self.migrations_dir))

            return {"status": "initialized", "migrations_dir": str(self.migrations_dir)}

        return {"status": "already_exists", "migrations_dir": str(self.migrations_dir)}

    async def create_migration(
        self, message: str, autogenerate: bool = True
    ) -> dict[str, Any]:
        """新しいマイグレーションファイルを作成"""
        config = self._get_config()

        # マイグレーションファイルを作成
        revision = command.revision(config, message=message, autogenerate=autogenerate)

        return {
            "revision": revision.revision,
            "message": message,
            "path": revision.path,
        }

    async def run_migrations(
        self, revision: str = "head", direction: str = "up"
    ) -> dict[str, Any]:
        """マイグレーションを実行"""
        config = self._get_config()

        try:
            if direction == "up":
                command.upgrade(config, revision)
            elif direction == "down":
                command.downgrade(config, revision)
            else:
                raise ValueError(f"Invalid direction: {direction}")

            return {"status": "success", "revision": revision, "direction": direction}
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "revision": revision,
                "direction": direction,
            }

    async def get_current_revision(self) -> str | None:
        """現在のリビジョンを取得"""
        config = self._get_config()

        # データベースへの接続を作成
        engine = create_engine(self.database_url)

        try:
            with engine.connect() as connection:
                context = MigrationContext.configure(connection)
                current_rev = context.get_current_revision()
                return current_rev
        finally:
            engine.dispose()

    async def get_migration_history(self) -> list[dict[str, Any]]:
        """マイグレーション履歴を取得"""
        config = self._get_config()
        script = ScriptDirectory.from_config(config)

        history = []
        for revision in script.walk_revisions():
            history.append(
                {
                    "revision": revision.revision,
                    "down_revision": revision.down_revision,
                    "description": revision.doc,
                    "path": revision.path,
                }
            )

        return history

    async def is_up_to_date(self) -> bool:
        """マイグレーションが最新かチェック"""
        current_rev = await self.get_current_revision()

        config = self._get_config()
        script = ScriptDirectory.from_config(config)
        head_revision = script.get_current_head()

        return current_rev == head_revision

    async def validate_migrations(self) -> dict[str, Any]:
        """マイグレーションの整合性を検証"""
        try:
            config = self._get_config()
            script = ScriptDirectory.from_config(config)

            # スクリプトディレクトリの検証
            script.get_revisions("head")

            return {"status": "valid", "message": "All migrations are valid"}
        except Exception as e:
            return {"status": "invalid", "error": str(e)}


def create_alembic_config(database_url: str, migrations_dir: Path) -> Config:
    """Alembic設定を作成"""
    config = Config()

    # データベースURL設定
    config.set_main_option("sqlalchemy.url", database_url)

    # マイグレーションディレクトリ設定
    config.set_main_option("script_location", str(migrations_dir))

    # その他の設定
    config.set_main_option("timezone", "UTC")
    config.set_main_option("truncate_slug_length", "40")

    return config


def create_migration(
    config: Config, message: str, autogenerate: bool = True
) -> dict[str, Any]:
    """マイグレーションファイルを作成（ユーティリティ関数）"""
    revision = command.revision(config, message=message, autogenerate=autogenerate)

    return {"revision": revision.revision, "message": message, "path": revision.path}


def run_migrations(config: Config, revision: str = "head") -> None:
    """マイグレーションを実行（ユーティリティ関数）"""
    command.upgrade(config, revision)


# 非同期コンテキストマネージャー
class AsyncMigrationManager:
    """非同期マイグレーション管理"""

    def __init__(self, database_url: str, migrations_dir: Path | None = None):
        self.manager = MigrationManager(database_url, migrations_dir)

    async def __aenter__(self):
        return self.manager

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # クリーンアップ処理（必要に応じて）
        pass
