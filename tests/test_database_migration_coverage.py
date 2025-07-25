"""
Comprehensive test coverage for app/database/migration.py to achieve 100% coverage.
This file focuses on covering all missing lines identified in the coverage report.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from pathlib import Path
from sqlalchemy.engine import Engine
from alembic.config import Config
from alembic.script import Script

from app.database.migration import (
    MigrationManager,
    create_alembic_config,
    create_migration,
    run_migrations,
    AsyncMigrationManager,
)


@pytest.fixture
def mock_database_url():
    """Mock database URL."""
    return "postgresql://test:test@localhost/testdb"


@pytest.fixture
def mock_migrations_dir(tmp_path):
    """Mock migrations directory."""
    return tmp_path / "migrations"


@pytest.fixture
def migration_manager(mock_database_url, mock_migrations_dir):
    """Create a migration manager instance."""
    return MigrationManager(mock_database_url, mock_migrations_dir)


class TestMigrationManager:
    """Test MigrationManager class."""

    def test_init(self, mock_database_url, mock_migrations_dir):
        """Test initialization of MigrationManager."""
        manager = MigrationManager(mock_database_url, mock_migrations_dir)
        assert manager.database_url == mock_database_url
        assert manager.migrations_dir == mock_migrations_dir
        assert manager.config is None

    def test_init_default_migrations_dir(self, mock_database_url):
        """Test initialization with default migrations directory."""
        manager = MigrationManager(mock_database_url)
        assert manager.migrations_dir == Path("migrations")

    def test_get_config(self, migration_manager):
        """Test _get_config method."""
        with patch("app.database.migration.create_alembic_config") as mock_create:
            mock_config = Mock(spec=Config)
            mock_create.return_value = mock_config

            # First call should create config
            config1 = migration_manager._get_config()
            assert config1 == mock_config
            assert migration_manager.config == mock_config
            mock_create.assert_called_once()

            # Second call should return cached config
            config2 = migration_manager._get_config()
            assert config2 == mock_config
            mock_create.assert_called_once()  # Still only called once

    @pytest.mark.asyncio
    async def test_initialize_migrations_new(self, migration_manager, mock_migrations_dir):
        """Test initializing new migrations."""
        # Ensure directory doesn't exist
        assert not mock_migrations_dir.exists()

        with patch("app.database.migration.command.init") as mock_init:
            with patch.object(migration_manager, "_get_config") as mock_get_config:
                mock_config = Mock()
                mock_get_config.return_value = mock_config

                result = await migration_manager.initialize_migrations()

                assert result["status"] == "initialized"
                assert result["migrations_dir"] == str(mock_migrations_dir)
                assert mock_migrations_dir.exists()
                mock_init.assert_called_once_with(mock_config, str(mock_migrations_dir))

    @pytest.mark.asyncio
    async def test_initialize_migrations_existing(self, migration_manager, mock_migrations_dir):
        """Test initializing when migrations already exist."""
        # Create the directory first
        mock_migrations_dir.mkdir(parents=True)

        with patch("app.database.migration.command.init") as mock_init:
            result = await migration_manager.initialize_migrations()

            assert result["status"] == "already_exists"
            assert result["migrations_dir"] == str(mock_migrations_dir)
            mock_init.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_migration_success(self, migration_manager):
        """Test successful migration creation."""
        mock_revision = Mock()
        mock_revision.revision = "abc123"
        mock_revision.path = "/path/to/migration.py"

        with patch("app.database.migration.command.revision") as mock_command:
            mock_command.return_value = mock_revision
            with patch.object(migration_manager, "_get_config") as mock_get_config:
                mock_config = Mock()
                mock_get_config.return_value = mock_config

                result = await migration_manager.create_migration("Test migration")

                assert result["revision"] == "abc123"
                assert result["message"] == "Test migration"
                assert result["path"] == "/path/to/migration.py"
                mock_command.assert_called_once_with(
                    mock_config, message="Test migration", autogenerate=True
                )

    @pytest.mark.asyncio
    async def test_create_migration_failure_none(self, migration_manager):
        """Test migration creation when revision returns None."""
        with patch("app.database.migration.command.revision") as mock_command:
            mock_command.return_value = None
            with patch.object(migration_manager, "_get_config") as mock_get_config:
                mock_config = Mock()
                mock_get_config.return_value = mock_config

                with pytest.raises(RuntimeError, match="Failed to create migration"):
                    await migration_manager.create_migration("Test migration")

    @pytest.mark.asyncio
    async def test_create_migration_unexpected_type(self, migration_manager):
        """Test migration creation with unexpected revision type."""
        mock_revision = Mock()
        # Missing required attributes
        del mock_revision.revision

        with patch("app.database.migration.command.revision") as mock_command:
            mock_command.return_value = mock_revision
            with patch.object(migration_manager, "_get_config") as mock_get_config:
                mock_config = Mock()
                mock_get_config.return_value = mock_config

                with pytest.raises(RuntimeError, match="Unexpected revision type"):
                    await migration_manager.create_migration("Test migration")

    @pytest.mark.asyncio
    async def test_run_migrations_upgrade_success(self, migration_manager):
        """Test successful upgrade migration."""
        with patch("app.database.migration.command.upgrade") as mock_upgrade:
            with patch.object(migration_manager, "_get_config") as mock_get_config:
                mock_config = Mock()
                mock_get_config.return_value = mock_config

                result = await migration_manager.run_migrations("head", "up")

                assert result["status"] == "success"
                assert result["revision"] == "head"
                assert result["direction"] == "up"
                mock_upgrade.assert_called_once_with(mock_config, "head")

    @pytest.mark.asyncio
    async def test_run_migrations_downgrade_success(self, migration_manager):
        """Test successful downgrade migration."""
        with patch("app.database.migration.command.downgrade") as mock_downgrade:
            with patch.object(migration_manager, "_get_config") as mock_get_config:
                mock_config = Mock()
                mock_get_config.return_value = mock_config

                result = await migration_manager.run_migrations("-1", "down")

                assert result["status"] == "success"
                assert result["revision"] == "-1"
                assert result["direction"] == "down"
                mock_downgrade.assert_called_once_with(mock_config, "-1")

    @pytest.mark.asyncio
    async def test_run_migrations_invalid_direction(self, migration_manager):
        """Test migration with invalid direction."""
        with patch.object(migration_manager, "_get_config") as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config

            result = await migration_manager.run_migrations("head", "sideways")

            assert result["status"] == "error"
            assert "Invalid direction: sideways" in result["error"]

    @pytest.mark.asyncio
    async def test_run_migrations_exception(self, migration_manager):
        """Test migration with exception during execution."""
        with patch("app.database.migration.command.upgrade") as mock_upgrade:
            mock_upgrade.side_effect = Exception("Database error")
            with patch.object(migration_manager, "_get_config") as mock_get_config:
                mock_config = Mock()
                mock_get_config.return_value = mock_config

                result = await migration_manager.run_migrations("head", "up")

                assert result["status"] == "error"
                assert "Database error" in result["error"]

    @pytest.mark.asyncio
    async def test_get_current_revision(self, migration_manager):
        """Test getting current revision."""
        mock_engine = Mock(spec=Engine)
        mock_connection = Mock()
        mock_context = Mock()
        mock_context.get_current_revision.return_value = "abc123"

        with patch("app.database.migration.create_engine") as mock_create_engine:
            mock_create_engine.return_value = mock_engine
            mock_engine.connect.return_value.__enter__.return_value = mock_connection
            
            with patch("app.database.migration.MigrationContext.configure") as mock_configure:
                mock_configure.return_value = mock_context

                revision = await migration_manager.get_current_revision()

                assert revision == "abc123"
                mock_create_engine.assert_called_once_with(migration_manager.database_url)
                mock_engine.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_migration_history(self, migration_manager):
        """Test getting migration history."""
        # Create mock revisions
        mock_rev1 = Mock()
        mock_rev1.revision = "rev1"
        mock_rev1.down_revision = None
        mock_rev1.doc = "Initial migration"
        mock_rev1.path = "/path/to/rev1.py"

        mock_rev2 = Mock()
        mock_rev2.revision = "rev2"
        mock_rev2.down_revision = "rev1"
        mock_rev2.doc = "Second migration"
        mock_rev2.path = "/path/to/rev2.py"

        mock_script = Mock()
        mock_script.walk_revisions.return_value = [mock_rev2, mock_rev1]

        with patch.object(migration_manager, "_get_config") as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            
            with patch("app.database.migration.ScriptDirectory.from_config") as mock_from_config:
                mock_from_config.return_value = mock_script

                history = await migration_manager.get_migration_history()

                assert len(history) == 2
                assert history[0]["revision"] == "rev2"
                assert history[0]["down_revision"] == "rev1"
                assert history[1]["revision"] == "rev1"
                assert history[1]["down_revision"] is None

    @pytest.mark.asyncio
    async def test_is_up_to_date_true(self, migration_manager):
        """Test when migrations are up to date."""
        with patch.object(migration_manager, "get_current_revision") as mock_get_current:
            mock_get_current.return_value = "head123"
            
            with patch.object(migration_manager, "_get_config") as mock_get_config:
                mock_config = Mock()
                mock_get_config.return_value = mock_config
                
                mock_script = Mock()
                mock_script.get_current_head.return_value = "head123"
                
                with patch("app.database.migration.ScriptDirectory.from_config") as mock_from_config:
                    mock_from_config.return_value = mock_script

                    is_current = await migration_manager.is_up_to_date()

                    assert is_current is True

    @pytest.mark.asyncio
    async def test_is_up_to_date_false(self, migration_manager):
        """Test when migrations are not up to date."""
        with patch.object(migration_manager, "get_current_revision") as mock_get_current:
            mock_get_current.return_value = "old123"
            
            with patch.object(migration_manager, "_get_config") as mock_get_config:
                mock_config = Mock()
                mock_get_config.return_value = mock_config
                
                mock_script = Mock()
                mock_script.get_current_head.return_value = "head123"
                
                with patch("app.database.migration.ScriptDirectory.from_config") as mock_from_config:
                    mock_from_config.return_value = mock_script

                    is_current = await migration_manager.is_up_to_date()

                    assert is_current is False

    @pytest.mark.asyncio
    async def test_validate_migrations_valid(self, migration_manager):
        """Test validating migrations when valid."""
        with patch.object(migration_manager, "_get_config") as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            
            mock_script = Mock()
            mock_script.get_revisions.return_value = ["rev1", "rev2"]
            
            with patch("app.database.migration.ScriptDirectory.from_config") as mock_from_config:
                mock_from_config.return_value = mock_script

                result = await migration_manager.validate_migrations()

                assert result["status"] == "valid"
                assert result["message"] == "All migrations are valid"

    @pytest.mark.asyncio
    async def test_validate_migrations_invalid(self, migration_manager):
        """Test validating migrations when invalid."""
        with patch.object(migration_manager, "_get_config") as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config
            
            with patch("app.database.migration.ScriptDirectory.from_config") as mock_from_config:
                mock_from_config.side_effect = Exception("Invalid migration structure")

                result = await migration_manager.validate_migrations()

                assert result["status"] == "invalid"
                assert "Invalid migration structure" in result["error"]


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_alembic_config(self, mock_database_url, mock_migrations_dir):
        """Test creating Alembic configuration."""
        config = create_alembic_config(mock_database_url, mock_migrations_dir)

        assert isinstance(config, Config)
        assert config.get_main_option("sqlalchemy.url") == mock_database_url
        assert config.get_main_option("script_location") == str(mock_migrations_dir)
        assert config.get_main_option("timezone") == "UTC"
        assert config.get_main_option("truncate_slug_length") == "40"

    def test_create_migration_utility_success(self):
        """Test create_migration utility function success."""
        mock_config = Mock(spec=Config)
        mock_revision = Mock()
        mock_revision.revision = "xyz789"
        mock_revision.path = "/path/to/xyz789.py"

        with patch("app.database.migration.command.revision") as mock_command:
            mock_command.return_value = mock_revision

            result = create_migration(mock_config, "Utility migration", autogenerate=False)

            assert result["revision"] == "xyz789"
            assert result["message"] == "Utility migration"
            assert result["path"] == "/path/to/xyz789.py"
            mock_command.assert_called_once_with(
                mock_config, message="Utility migration", autogenerate=False
            )

    def test_create_migration_utility_failure_none(self):
        """Test create_migration utility when revision is None."""
        mock_config = Mock(spec=Config)

        with patch("app.database.migration.command.revision") as mock_command:
            mock_command.return_value = None

            with pytest.raises(RuntimeError, match="Failed to create migration"):
                create_migration(mock_config, "Failed migration")

    def test_create_migration_utility_unexpected_type(self):
        """Test create_migration utility with unexpected type."""
        mock_config = Mock(spec=Config)
        mock_revision = "unexpected_string"  # Wrong type

        with patch("app.database.migration.command.revision") as mock_command:
            mock_command.return_value = mock_revision

            with pytest.raises(RuntimeError, match="Unexpected revision type"):
                create_migration(mock_config, "Bad migration")

    def test_run_migrations_utility(self):
        """Test run_migrations utility function."""
        mock_config = Mock(spec=Config)

        with patch("app.database.migration.command.upgrade") as mock_upgrade:
            run_migrations(mock_config, "custom_revision")

            mock_upgrade.assert_called_once_with(mock_config, "custom_revision")


class TestAsyncMigrationManager:
    """Test AsyncMigrationManager context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_database_url, mock_migrations_dir):
        """Test async context manager functionality."""
        async with AsyncMigrationManager(mock_database_url, mock_migrations_dir) as manager:
            assert isinstance(manager, MigrationManager)
            assert manager.database_url == mock_database_url
            assert manager.migrations_dir == mock_migrations_dir

    @pytest.mark.asyncio
    async def test_async_context_manager_exit(self, mock_database_url):
        """Test async context manager exit."""
        manager = AsyncMigrationManager(mock_database_url)
        
        # Test __aenter__ and __aexit__ directly
        entered_manager = await manager.__aenter__()
        assert isinstance(entered_manager, MigrationManager)
        
        # Should not raise any exceptions
        await manager.__aexit__(None, None, None)
        
        # Test with exception info
        await manager.__aexit__(Exception, Exception("test"), None)