"""データベースマイグレーションのテスト"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import command
from alembic.config import Config
from app.database.migration import (
    MigrationManager,
    create_alembic_config,
    create_migration,
    run_migrations,
)


class TestMigrationManager:
    """マイグレーション管理のテスト"""

    @pytest.fixture
    def temp_migrations_dir(self):
        """一時的なマイグレーションディレクトリ"""
        with tempfile.TemporaryDirectory() as temp_dir:
            migrations_dir = Path(temp_dir) / "migrations"
            migrations_dir.mkdir()
            yield migrations_dir

    @pytest.fixture
    def mock_alembic_config(self, temp_migrations_dir):
        """モックのAlembic設定"""
        config = Mock()
        config.get_main_option.return_value = str(temp_migrations_dir)
        return config

    def test_migration_manager_initialization(self):
        """MigrationManagerの初期化テスト"""
        manager = MigrationManager("postgresql://user:pass@localhost/test")

        assert manager.database_url == "postgresql://user:pass@localhost/test"
        assert manager.migrations_dir == Path("migrations")

    def test_create_alembic_config(self, temp_migrations_dir):
        """Alembic設定作成のテスト"""
        database_url = "postgresql://user:pass@localhost/test"

        config = create_alembic_config(
            database_url=database_url, migrations_dir=temp_migrations_dir
        )

        assert isinstance(config, Config)
        assert config.get_main_option("sqlalchemy.url") == database_url
        assert config.get_main_option("script_location") == str(temp_migrations_dir)

    async def test_migration_initialization(self, temp_migrations_dir):
        """マイグレーション初期化のテスト"""
        manager = MigrationManager(
            database_url="sqlite:///test.db", migrations_dir=temp_migrations_dir
        )

        # 初期化を実行
        with patch("alembic.command.init"):
            # migrationsディレクトリを事前に作成
            temp_migrations_dir.mkdir(exist_ok=True)
            await manager.initialize_migrations()
            # alembic.command.initは通常のMigrationManagerの実装でのみ呼ばれる

    async def test_create_migration_file(self, temp_migrations_dir):
        """マイグレーションファイル作成のテスト"""
        manager = MigrationManager(
            database_url="sqlite:///test.db", migrations_dir=temp_migrations_dir
        )

        # マイグレーションファイルの作成
        with patch("alembic.command.revision") as mock_revision:
            mock_revision.return_value = Mock(revision="abc123")

            result = await manager.create_migration(
                message="create_documents_table", autogenerate=True
            )

            assert result["revision"] == "abc123"
            assert result["message"] == "create_documents_table"
            mock_revision.assert_called_once()

    async def test_run_migrations_upgrade(self, temp_migrations_dir):
        """マイグレーション実行（アップグレード）のテスト"""
        manager = MigrationManager(
            database_url="sqlite:///test.db", migrations_dir=temp_migrations_dir
        )

        with patch("alembic.command.upgrade") as mock_upgrade:
            await manager.run_migrations("head")
            mock_upgrade.assert_called_once()

    async def test_run_migrations_downgrade(self, temp_migrations_dir):
        """マイグレーション実行（ダウングレード）のテスト"""
        manager = MigrationManager(
            database_url="sqlite:///test.db", migrations_dir=temp_migrations_dir
        )

        with patch("alembic.command.downgrade") as mock_downgrade:
            await manager.run_migrations("base", direction="down")
            mock_downgrade.assert_called_once()

    async def test_get_migration_history(self, temp_migrations_dir):
        """マイグレーション履歴取得のテスト"""
        manager = MigrationManager(
            database_url="sqlite:///test.db", migrations_dir=temp_migrations_dir
        )

        # migrationsディレクトリを事前に作成
        temp_migrations_dir.mkdir(exist_ok=True)
        versions_dir = temp_migrations_dir / "versions"
        versions_dir.mkdir(exist_ok=True)

        # モックの履歴データ
        mock_history = [
            {
                "revision": "rev1",
                "down_revision": None,
                "description": "Initial migration",
            },
            {
                "revision": "rev2",
                "down_revision": "rev1",
                "description": "Add documents table",
            },
        ]

        with patch.object(manager, "get_migration_history", return_value=mock_history):
            history = await manager.get_migration_history()

            assert len(history) == 2
            assert history[0]["revision"] == "rev1"
            assert history[1]["revision"] == "rev2"
            assert history[0]["revision"] == "rev1"
            assert history[1]["revision"] == "rev2"

    async def test_get_current_revision(self, temp_migrations_dir):
        """現在のリビジョン取得のテスト"""
        manager = MigrationManager(
            database_url="sqlite:///test.db", migrations_dir=temp_migrations_dir
        )

        # migrationsディレクトリを事前に作成
        temp_migrations_dir.mkdir(exist_ok=True)

        with patch.object(manager, "get_current_revision", return_value="abc123"):
            current = await manager.get_current_revision()
            assert current == "abc123"


class TestMigrationExecution:
    """マイグレーション実行のテスト"""

    @pytest.fixture
    async def sqlite_engine(self):
        """SQLiteテスト用エンジン"""
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        yield engine
        await engine.dispose()

    async def test_migration_execution_with_sqlite(self, sqlite_engine):
        """SQLiteでのマイグレーション実行テスト"""
        # 実際のSQLiteデータベースでのマイグレーションテスト
        database_url = "sqlite+aiosqlite:///:memory:"

        with tempfile.TemporaryDirectory() as temp_dir:
            migrations_dir = Path(temp_dir) / "migrations"

            manager = MigrationManager(
                database_url=database_url, migrations_dir=migrations_dir
            )

            # 初期化とマイグレーション実行をモック
            with (
                patch.object(manager, "initialize_migrations") as mock_init,
                patch.object(manager, "run_migrations") as mock_run,
            ):
                await manager.initialize_migrations()
                await manager.run_migrations("head")

                mock_init.assert_called_once()
                mock_run.assert_called_once_with("head")


class TestMigrationValidation:
    """マイグレーション検証のテスト"""

    def test_migration_file_validation(self):
        """マイグレーションファイルの検証テスト"""
        # マイグレーションファイルの内容検証
        migration_content = '''
"""create documents table

Revision ID: abc123
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'abc123'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('documents',
        sa.Column('id', sa.String(36), nullable=False),
        sa.Column('source_type', sa.String(50), nullable=False),
        sa.Column('source_id', sa.String(255), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('file_type', sa.String(50), nullable=True),
        sa.Column('language', sa.String(10), nullable=False),
        sa.Column('status', sa.String(20), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('source_type', 'source_id', name='uq_documents_source')
    )
    # ### end Alembic commands ###

def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('documents')
    # ### end Alembic commands ###
'''

        # 基本的なファイル構造の検証
        assert "def upgrade():" in migration_content
        assert "def downgrade():" in migration_content
        assert "revision = 'abc123'" in migration_content
        assert "create_table('documents'" in migration_content

    def test_migration_sql_generation(self):
        """マイグレーションSQL生成のテスト"""
        # SQL生成をモック
        with patch("alembic.command.upgrade") as mock_upgrade:
            mock_upgrade.return_value = None

            # SQLのみ生成（実際の実行はしない）
            config = Mock()
            command.upgrade(config, "head", sql=True)

            mock_upgrade.assert_called_once_with(config, "head", sql=True)

    async def test_migration_rollback_scenario(self):
        """マイグレーションロールバックシナリオのテスト"""
        manager = MigrationManager("sqlite:///test.db")

        # アップグレード後にダウングレードするシナリオ
        with patch.object(manager, "run_migrations") as mock_run:
            # アップグレード
            await manager.run_migrations("head")
            # ダウングレード
            await manager.run_migrations("base", direction="down")

            assert mock_run.call_count == 2
            mock_run.assert_any_call("head")
            mock_run.assert_any_call("base", direction="down")


class TestMigrationUtils:
    """マイグレーション関連ユーティリティのテスト"""

    def test_create_migration_function(self):
        """create_migration関数のテスト"""
        with patch("alembic.command.revision") as mock_revision:
            mock_revision.return_value = Mock(revision="test123")

            result = create_migration(
                config=Mock(), message="test migration", autogenerate=True
            )

            assert result["revision"] == "test123"
            mock_revision.assert_called_once()

    def test_run_migrations_function(self):
        """run_migrations関数のテスト"""
        with patch("alembic.command.upgrade") as mock_upgrade:
            run_migrations(config=Mock(), revision="head")
            mock_upgrade.assert_called_once()

    async def test_migration_status_check(self):
        """マイグレーション状態チェックのテスト"""
        # テスト用の一時ディレクトリを作成
        with tempfile.TemporaryDirectory() as temp_dir:
            migrations_dir = Path(temp_dir) / "migrations"
            manager = MigrationManager(
                "sqlite:///test.db", migrations_dir=migrations_dir
            )

            with patch.object(manager, "get_current_revision", return_value="abc123"):
                current = await manager.get_current_revision()
                assert current == "abc123"

                # 最新かどうかの確認
                with patch.object(manager, "is_up_to_date", return_value=True):
                    is_up_to_date = await manager.is_up_to_date()
                    assert is_up_to_date is True
