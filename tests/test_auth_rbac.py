"""RBAC権限管理のテスト"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class TestRBACSystem:
    """RBAC権限管理システムのテストクラス"""

    def test_role_definition(self):
        """ロール定義のテスト"""
        from app.core.auth import Permission, Role

        # 管理者ロール
        admin_role = Role(
            name="admin",
            permissions=[
                Permission.READ,
                Permission.WRITE,
                Permission.DELETE,
                Permission.ADMIN,
            ],
        )

        assert admin_role.name == "admin"
        assert Permission.READ in admin_role.permissions
        assert Permission.ADMIN in admin_role.permissions
        assert len(admin_role.permissions) == 4

    def test_permission_check(self):
        """権限チェックのテスト"""
        from app.core.auth import Permission, has_permission

        user_permissions = [Permission.READ, Permission.WRITE]

        # 持っている権限
        assert has_permission(user_permissions, Permission.READ) is True
        assert has_permission(user_permissions, Permission.WRITE) is True

        # 持っていない権限
        assert has_permission(user_permissions, Permission.DELETE) is False
        assert has_permission(user_permissions, Permission.ADMIN) is False

    def test_role_hierarchy(self):
        """ロール階層のテスト"""
        from app.core.auth import Permission, Role, get_effective_permissions

        # 基本ユーザーロール
        user_role = Role("user", [Permission.READ])

        # エディターロール（ユーザー権限を継承）
        editor_role = Role("editor", [Permission.WRITE], parent=user_role)

        # 管理者ロール（エディター権限を継承）
        admin_role = Role(
            "admin", [Permission.DELETE, Permission.ADMIN], parent=editor_role
        )

        # 有効権限の確認
        admin_permissions = get_effective_permissions(admin_role)

        assert Permission.READ in admin_permissions  # ユーザーから継承
        assert Permission.WRITE in admin_permissions  # エディターから継承
        assert Permission.DELETE in admin_permissions  # 管理者固有
        assert Permission.ADMIN in admin_permissions  # 管理者固有

    def test_resource_based_permissions(self):
        """リソースベース権限のテスト"""
        from app.core.auth import ResourcePermission, check_resource_permission

        # ドキュメントリソースの権限
        doc_permission = ResourcePermission(
            resource_type="document",
            resource_id="doc123",
            permissions=["read", "write"],
        )

        # 権限チェック
        assert (
            check_resource_permission(doc_permission, "document", "doc123", "read")
            is True
        )
        assert (
            check_resource_permission(doc_permission, "document", "doc123", "write")
            is True
        )
        assert (
            check_resource_permission(doc_permission, "document", "doc123", "delete")
            is False
        )
        assert (
            check_resource_permission(doc_permission, "document", "doc456", "read")
            is False
        )


class TestRBACAPI:
    """RBAC APIのテストクラス"""

    def test_admin_access_to_user_management(self):
        """管理者のユーザー管理アクセステスト"""
        client = TestClient(app)

        # 管理者でログイン
        login_data = {"username": "admin@example.com", "password": "adminpassword"}
        login_response = client.post("/v1/auth/login", data=login_data)
        access_token = login_response.json()["access_token"]

        # ユーザー一覧取得（管理者権限必要）
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/v1/admin/users", headers=headers)

        assert response.status_code == 200

    def test_user_access_denied_to_admin_endpoints(self):
        """一般ユーザーの管理者エンドポイントアクセス拒否テスト"""
        client = TestClient(app)

        # 一般ユーザーでログイン
        login_data = {"username": "user@example.com", "password": "userpassword"}
        login_response = client.post("/v1/auth/login", data=login_data)
        access_token = login_response.json()["access_token"]

        # 管理者エンドポイントにアクセス試行
        headers = {"Authorization": f"Bearer {access_token}"}
        response = client.get("/v1/admin/users", headers=headers)

        assert response.status_code == 403
        data = response.json()
        assert data["error"]["code"] == "AUTHORIZATION_ERROR"

    def test_editor_document_permissions(self):
        """エディターのドキュメント権限テスト"""
        client = TestClient(app)

        # エディターでログイン
        login_data = {"username": "editor@example.com", "password": "editorpassword"}
        login_response = client.post("/v1/auth/login", data=login_data)
        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # ドキュメント読み取り（成功）
        response = client.get("/v1/documents", headers=headers)
        assert response.status_code == 200

        # ドキュメント作成（成功）
        document_data = {
            "title": "Test Document",
            "content": "Test content",
            "source_type": "test",
        }
        response = client.post("/v1/documents", json=document_data, headers=headers)
        assert response.status_code == 201

        # ドキュメント削除（失敗 - 管理者権限必要）
        response = client.delete("/v1/documents/123", headers=headers)
        assert response.status_code == 403

    def test_role_assignment_api(self):
        """ロール割り当てAPIのテスト"""
        client = TestClient(app)

        # 管理者でログイン
        login_data = {"username": "admin@example.com", "password": "adminpassword"}
        login_response = client.post("/v1/auth/login", data=login_data)
        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # ユーザーにロール割り当て
        role_data = {"user_id": "user123", "role": "editor"}
        response = client.post("/v1/admin/users/roles", json=role_data, headers=headers)

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Role assigned successfully"

    def test_permission_inheritance(self):
        """権限継承のテスト"""
        client = TestClient(app)

        # 階層ロールを持つユーザーでログイン
        login_data = {"username": "manager@example.com", "password": "managerpassword"}
        login_response = client.post("/v1/auth/login", data=login_data)
        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # 基本権限（読み取り）
        response = client.get("/v1/documents", headers=headers)
        assert response.status_code == 200

        # 継承された権限（書き込み）
        document_data = {
            "title": "Manager Document",
            "content": "Manager content",
            "source_type": "test",
        }
        response = client.post("/v1/documents", json=document_data, headers=headers)
        assert response.status_code == 201

        # マネージャー固有権限（チーム管理）
        response = client.get("/v1/admin/team", headers=headers)
        assert response.status_code == 200


class TestRBACMiddleware:
    """RBAC認証ミドルウェアのテストクラス"""

    def test_permission_decorator(self):
        """権限デコレーターのテスト"""
        from app.core.auth import Permission, require_permission

        @require_permission(Permission.ADMIN)
        def admin_only_function():
            return "admin access granted"

        # 管理者権限を持つユーザー
        with patch("app.core.auth.get_current_user_permissions") as mock_perms:
            mock_perms.return_value = [Permission.ADMIN, Permission.READ]
            result = admin_only_function()
            assert result == "admin access granted"

    def test_role_decorator(self):
        """ロールデコレーターのテスト"""
        from app.core.auth import require_role

        @require_role("admin")
        def admin_only_function():
            return "admin role access granted"

        # 管理者ロールを持つユーザー
        with patch("app.core.auth.get_current_user_role") as mock_role:
            mock_role.return_value = "admin"
            result = admin_only_function()
            assert result == "admin role access granted"

    def test_resource_permission_decorator(self):
        """リソース権限デコレーターのテスト"""
        from app.core.auth import require_resource_permission

        @require_resource_permission("document", "read")
        def read_document(doc_id: str):
            return f"reading document {doc_id}"

        # ドキュメント読み取り権限を持つユーザー
        with patch("app.core.auth.check_user_resource_permission") as mock_check:
            mock_check.return_value = True
            result = read_document("doc123")
            assert result == "reading document doc123"


class TestRBACDatabase:
    """RBAC データベース操作のテストクラス"""

    def test_user_role_storage(self):
        """ユーザーロール保存のテスト"""
        from app.core.auth import assign_role_to_user, get_user_roles

        user_id = "user123"
        role = "editor"

        # ロール割り当て
        assign_role_to_user(user_id, role)

        # ロール取得
        user_roles = get_user_roles(user_id)
        assert role in user_roles

    def test_role_permissions_storage(self):
        """ロール権限保存のテスト"""
        from app.core.auth import Permission, create_role, get_role_permissions

        role_name = "custom_role"
        permissions = [Permission.READ, Permission.WRITE]

        # カスタムロール作成
        create_role(role_name, permissions)

        # ロール権限取得
        stored_permissions = get_role_permissions(role_name)
        assert Permission.READ in stored_permissions
        assert Permission.WRITE in stored_permissions

    def test_resource_permissions_storage(self):
        """リソース権限保存のテスト"""
        from app.core.auth import (
            check_user_resource_permission,
            grant_resource_permission,
        )

        user_id = "user123"
        resource_type = "document"
        resource_id = "doc123"
        permission = "read"

        # リソース権限付与
        grant_resource_permission(user_id, resource_type, resource_id, permission)

        # リソース権限確認
        has_permission = check_user_resource_permission(
            user_id, resource_type, resource_id, permission
        )
        assert has_permission is True


class TestRBACIntegration:
    """RBAC統合テスト"""

    def test_complete_authorization_flow(self):
        """完全な認可フローのテスト"""
        client = TestClient(app)

        # 1. ユーザー登録
        register_data = {
            "email": "newuser@example.com",
            "password": "newpassword",
            "role": "user",
        }
        response = client.post("/v1/auth/register", json=register_data)
        assert response.status_code == 201

        # 2. ログイン
        login_data = {"username": "newuser@example.com", "password": "newpassword"}
        login_response = client.post("/v1/auth/login", data=login_data)
        access_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # 3. 基本権限でのアクセス
        response = client.get("/v1/documents", headers=headers)
        assert response.status_code == 200

        # 4. 権限不足でのアクセス拒否
        response = client.delete("/v1/documents/123", headers=headers)
        assert response.status_code == 403

        # 5. 管理者がロール変更
        admin_login = {"username": "admin@example.com", "password": "adminpassword"}
        admin_response = client.post("/v1/auth/login", data=admin_login)
        admin_token = admin_response.json()["access_token"]
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        role_change = {"user_email": "newuser@example.com", "role": "editor"}
        response = client.put(
            "/v1/admin/users/role", json=role_change, headers=admin_headers
        )
        assert response.status_code == 200

        # 6. 新しい権限でのアクセス
        document_data = {
            "title": "New User Document",
            "content": "Content from upgraded user",
            "source_type": "test",
        }
        response = client.post("/v1/documents", json=document_data, headers=headers)
        assert response.status_code == 201
