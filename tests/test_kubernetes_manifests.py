"""Kubernetesマニフェストテスト

TDD実装：マニフェストのテスト/検証
- マニフェスト構文検証: YAML形式、必須フィールド、リソース制限
- セキュリティ検証: セキュリティコンテキスト、RBAC、ネットワークポリシー
- 設定検証: ConfigMap、Secret、環境変数の整合性
- ヘルスチェック検証: Liveness/Readiness Probe設定
- リソース検証: CPU/メモリ制限、ストレージ要件
"""

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.deployment.kubernetes_validator import (
    KubernetesManifestValidator,
    KubernetesValidationConfig,
    ManifestValidationError,
    ValidationSeverity,
)


@pytest.fixture
def validation_config() -> KubernetesValidationConfig:
    """バリデーション設定"""
    return KubernetesValidationConfig(
        enable_security_validation=True,
        enable_resource_validation=True,
        enable_health_check_validation=True,
        strict_mode=True,
        required_labels=["app", "version", "environment"],
        required_annotations=["prometheus.io/scrape"],
        max_cpu_limit="2",
        max_memory_limit="4Gi",
        min_replicas=2,
        max_replicas=10,
    )


@pytest.fixture
def sample_deployment_manifest() -> dict[str, Any]:
    """サンプルDeploymentマニフェスト"""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": "spec-rag-api",
            "namespace": "spec-rag",
            "labels": {
                "app": "spec-rag-api",
                "version": "v1.0.0",
                "environment": "production",
                "component": "api",
            },
            "annotations": {
                "prometheus.io/scrape": "true",
                "prometheus.io/port": "8000",
                "deployment.kubernetes.io/revision": "1",
            },
        },
        "spec": {
            "replicas": 3,
            "selector": {
                "matchLabels": {
                    "app": "spec-rag-api",
                    "version": "v1.0.0",
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": "spec-rag-api",
                        "version": "v1.0.0",
                        "environment": "production",
                    }
                },
                "spec": {
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "fsGroup": 1000,
                    },
                    "containers": [
                        {
                            "name": "api",
                            "image": "spec-rag-api:v1.0.0",
                            "imagePullPolicy": "IfNotPresent",
                            "ports": [
                                {
                                    "containerPort": 8000,
                                    "name": "http",
                                    "protocol": "TCP",
                                }
                            ],
                            "env": [
                                {
                                    "name": "DATABASE_URL",
                                    "valueFrom": {
                                        "secretKeyRef": {
                                            "name": "spec-rag-secrets",
                                            "key": "database-url",
                                        }
                                    },
                                },
                                {
                                    "name": "REDIS_URL",
                                    "valueFrom": {
                                        "configMapKeyRef": {
                                            "name": "spec-rag-config",
                                            "key": "redis-url",
                                        }
                                    },
                                },
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "500m",
                                    "memory": "1Gi",
                                },
                                "limits": {
                                    "cpu": "1",
                                    "memory": "2Gi",
                                },
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000,
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3,
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8000,
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5,
                                "timeoutSeconds": 3,
                                "failureThreshold": 2,
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "capabilities": {
                                    "drop": ["ALL"]
                                },
                            },
                        }
                    ],
                },
            },
        },
    }


@pytest.fixture
def sample_service_manifest() -> dict[str, Any]:
    """サンプルServiceマニフェスト"""
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": "spec-rag-api-service",
            "namespace": "spec-rag",
            "labels": {
                "app": "spec-rag-api",
                "version": "v1.0.0",
                "environment": "production",
            },
            "annotations": {
                "prometheus.io/scrape": "true",
                "service.beta.kubernetes.io/aws-load-balancer-type": "nlb",
            },
        },
        "spec": {
            "type": "ClusterIP",
            "ports": [
                {
                    "port": 80,
                    "targetPort": 8000,
                    "protocol": "TCP",
                    "name": "http",
                }
            ],
            "selector": {
                "app": "spec-rag-api",
                "version": "v1.0.0",
            },
        },
    }


@pytest.fixture
def sample_configmap_manifest() -> dict[str, Any]:
    """サンプルConfigMapマニフェスト"""
    return {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {
            "name": "spec-rag-config",
            "namespace": "spec-rag",
            "labels": {
                "app": "spec-rag-api",
                "version": "v1.0.0",
                "environment": "production",
            },
            "annotations": {
                "prometheus.io/scrape": "false",
            },
        },
        "data": {
            "redis-url": "redis://spec-rag-redis:6379/0",
            "aperturedb-host": "spec-rag-aperturedb",
            "aperturedb-port": "55555",
            "log-level": "INFO",
            "workers": "4",
        },
    }


class TestKubernetesValidationConfig:
    """Kubernetes検証設定のテスト"""

    @pytest.mark.unit
    def test_config_creation(self):
        """設定の作成"""
        config = KubernetesValidationConfig(
            enable_security_validation=True,
            enable_resource_validation=True,
            strict_mode=False,
        )

        assert config.enable_security_validation is True
        assert config.enable_resource_validation is True
        assert config.strict_mode is False
        assert config.required_labels == ["app", "version"]  # デフォルト値

    @pytest.mark.unit
    def test_config_validation_success(self):
        """設定値のバリデーション（成功）"""
        config = KubernetesValidationConfig(
            max_cpu_limit="4",
            max_memory_limit="8Gi",
            min_replicas=1,
            max_replicas=20,
        )

        assert config.max_cpu_limit == "4"
        assert config.max_memory_limit == "8Gi"
        assert config.min_replicas == 1
        assert config.max_replicas == 20

    @pytest.mark.unit
    def test_config_validation_invalid_replicas(self):
        """無効なレプリカ数のバリデーション"""
        with pytest.raises(ValueError, match="min_replicas must be greater than 0"):
            KubernetesValidationConfig(
                min_replicas=0,
            )

        with pytest.raises(ValueError, match="max_replicas must be greater than min_replicas"):
            KubernetesValidationConfig(
                min_replicas=10,
                max_replicas=5,
            )


class TestManifestSyntaxValidation:
    """マニフェスト構文検証のテスト"""

    @pytest.mark.unit
    async def test_valid_deployment_manifest(
        self, validation_config: KubernetesValidationConfig, sample_deployment_manifest: dict[str, Any]
    ):
        """有効なDeploymentマニフェストの検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        result = await validator.validate_manifest(sample_deployment_manifest)

        assert result.is_valid is True
        assert result.severity == ValidationSeverity.INFO
        assert len(result.errors) == 0
        assert "Valid Deployment manifest" in result.message

    @pytest.mark.unit
    async def test_valid_service_manifest(
        self, validation_config: KubernetesValidationConfig, sample_service_manifest: dict[str, Any]
    ):
        """有効なServiceマニフェストの検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        result = await validator.validate_manifest(sample_service_manifest)

        assert result.is_valid is True
        assert result.severity == ValidationSeverity.INFO
        assert len(result.errors) == 0

    @pytest.mark.unit
    async def test_invalid_manifest_missing_required_fields(
        self, validation_config: KubernetesValidationConfig
    ):
        """必須フィールド欠如の検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        invalid_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            # metadataが欠如
            "spec": {
                "replicas": 1,
            },
        }

        result = await validator.validate_manifest(invalid_manifest)

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert len(result.errors) > 0
        assert any("metadata" in error for error in result.errors)

    @pytest.mark.unit
    async def test_invalid_manifest_wrong_api_version(
        self, validation_config: KubernetesValidationConfig
    ):
        """無効なapiVersionの検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        invalid_manifest = {
            "apiVersion": "invalid/v1",
            "kind": "Deployment",
            "metadata": {"name": "test"},
            "spec": {},
        }

        result = await validator.validate_manifest(invalid_manifest)

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert any("apiVersion" in error for error in result.errors)


class TestSecurityValidation:
    """セキュリティ検証のテスト"""

    @pytest.mark.unit
    async def test_security_context_validation_success(
        self, validation_config: KubernetesValidationConfig, sample_deployment_manifest: dict[str, Any]
    ):
        """セキュリティコンテキストの検証（成功）"""
        validator = KubernetesManifestValidator(config=validation_config)

        result = await validator.validate_security(sample_deployment_manifest)

        assert result.is_valid is True
        assert result.has_security_context is True
        assert result.runs_as_non_root is True
        assert result.read_only_root_filesystem is True

    @pytest.mark.unit
    async def test_security_context_validation_missing(
        self, validation_config: KubernetesValidationConfig
    ):
        """セキュリティコンテキスト欠如の検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        insecure_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "test"},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "app",
                                "image": "test:latest",
                                # securityContextが欠如
                            }
                        ]
                    }
                }
            },
        }

        result = await validator.validate_security(insecure_manifest)

        assert result.is_valid is False
        assert result.has_security_context is False
        assert len(result.security_violations) > 0

    @pytest.mark.unit
    async def test_privileged_container_detection(
        self, validation_config: KubernetesValidationConfig
    ):
        """特権コンテナの検出"""
        validator = KubernetesManifestValidator(config=validation_config)

        privileged_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "test"},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "app",
                                "image": "test:latest",
                                "securityContext": {
                                    "privileged": True,  # 特権コンテナ
                                },
                            }
                        ]
                    }
                }
            },
        }

        result = await validator.validate_security(privileged_manifest)

        assert result.is_valid is False
        assert result.has_privileged_containers is True
        assert "privileged containers detected" in str(result.security_violations).lower()


class TestResourceValidation:
    """リソース検証のテスト"""

    @pytest.mark.unit
    async def test_resource_limits_validation_success(
        self, validation_config: KubernetesValidationConfig, sample_deployment_manifest: dict[str, Any]
    ):
        """リソース制限の検証（成功）"""
        validator = KubernetesManifestValidator(config=validation_config)

        result = await validator.validate_resources(sample_deployment_manifest)

        assert result.is_valid is True
        assert result.has_resource_limits is True
        assert result.has_resource_requests is True
        assert len(result.resource_violations) == 0

    @pytest.mark.unit
    async def test_resource_limits_missing(
        self, validation_config: KubernetesValidationConfig
    ):
        """リソース制限欠如の検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        no_limits_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "test"},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "app",
                                "image": "test:latest",
                                # resourcesが欠如
                            }
                        ]
                    }
                }
            },
        }

        result = await validator.validate_resources(no_limits_manifest)

        assert result.is_valid is False
        assert result.has_resource_limits is False
        assert result.has_resource_requests is False
        assert len(result.resource_violations) > 0

    @pytest.mark.unit
    async def test_excessive_resource_limits(
        self, validation_config: KubernetesValidationConfig
    ):
        """過剰なリソース制限の検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        excessive_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "test"},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "app",
                                "image": "test:latest",
                                "resources": {
                                    "limits": {
                                        "cpu": "10",  # 制限値（2）を超過
                                        "memory": "16Gi",  # 制限値（4Gi）を超過
                                    }
                                },
                            }
                        ]
                    }
                }
            },
        }

        result = await validator.validate_resources(excessive_manifest)

        assert result.is_valid is False
        assert "exceeds maximum" in str(result.resource_violations).lower()


class TestHealthCheckValidation:
    """ヘルスチェック検証のテスト"""

    @pytest.mark.unit
    async def test_health_check_validation_success(
        self, validation_config: KubernetesValidationConfig, sample_deployment_manifest: dict[str, Any]
    ):
        """ヘルスチェックの検証（成功）"""
        validator = KubernetesManifestValidator(config=validation_config)

        result = await validator.validate_health_checks(sample_deployment_manifest)

        assert result.is_valid is True
        assert result.has_liveness_probe is True
        assert result.has_readiness_probe is True
        assert len(result.health_check_violations) == 0

    @pytest.mark.unit
    async def test_health_check_missing(
        self, validation_config: KubernetesValidationConfig
    ):
        """ヘルスチェック欠如の検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        no_health_check_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "test"},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "app",
                                "image": "test:latest",
                                # livenessProbe, readinessProbeが欠如
                            }
                        ]
                    }
                }
            },
        }

        result = await validator.validate_health_checks(no_health_check_manifest)

        assert result.is_valid is False
        assert result.has_liveness_probe is False
        assert result.has_readiness_probe is False
        assert len(result.health_check_violations) > 0

    @pytest.mark.unit
    async def test_health_check_configuration_validation(
        self, validation_config: KubernetesValidationConfig
    ):
        """ヘルスチェック設定の検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        invalid_health_check_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "test"},
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "app",
                                "image": "test:latest",
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": 8000,
                                    },
                                    "initialDelaySeconds": 1,  # 短すぎる
                                    "periodSeconds": 1,  # 短すぎる
                                    "timeoutSeconds": 30,  # 長すぎる
                                },
                            }
                        ]
                    }
                }
            },
        }

        result = await validator.validate_health_checks(invalid_health_check_manifest)

        assert result.is_valid is False
        # 具体的な検証エラーを確認
        violations = result.health_check_violations
        assert any("initialDelaySeconds too short" in violation for violation in violations)
        assert any("timeoutSeconds too long" in violation for violation in violations)


class TestLabelAndAnnotationValidation:
    """ラベル・アノテーション検証のテスト"""

    @pytest.mark.unit
    async def test_required_labels_validation_success(
        self, validation_config: KubernetesValidationConfig, sample_deployment_manifest: dict[str, Any]
    ):
        """必須ラベルの検証（成功）"""
        validator = KubernetesManifestValidator(config=validation_config)

        result = await validator.validate_labels_and_annotations(sample_deployment_manifest)

        assert result.is_valid is True
        assert result.has_required_labels is True
        assert result.has_required_annotations is True

    @pytest.mark.unit
    async def test_missing_required_labels(
        self, validation_config: KubernetesValidationConfig
    ):
        """必須ラベル欠如の検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        missing_labels_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "test",
                "labels": {
                    "app": "test",
                    # "version", "environment"が欠如
                },
            },
        }

        result = await validator.validate_labels_and_annotations(missing_labels_manifest)

        assert result.is_valid is False
        assert result.has_required_labels is False
        assert "missing required labels" in str(result.label_violations).lower()

    @pytest.mark.unit
    async def test_invalid_label_values(
        self, validation_config: KubernetesValidationConfig
    ):
        """無効なラベル値の検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        invalid_labels_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "test",
                "labels": {
                    "app": "Test_App!",  # 無効な文字
                    "version": "",  # 空の値
                    "environment": "prod",
                },
            },
        }

        result = await validator.validate_labels_and_annotations(invalid_labels_manifest)

        assert result.is_valid is False
        assert "invalid label format" in str(result.label_violations).lower()


class TestManifestFileValidation:
    """マニフェストファイル検証のテスト"""

    @pytest.mark.unit
    async def test_validate_yaml_file(
        self, validation_config: KubernetesValidationConfig, sample_deployment_manifest: dict[str, Any]
    ):
        """YAMLファイルの検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        # 一時ファイルに書き出し
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sample_deployment_manifest, f)
            temp_file = f.name

        try:
            result = await validator.validate_file(temp_file)

            assert result.is_valid is True
            assert result.file_path == temp_file
            assert len(result.validation_results) >= 1

        finally:
            os.unlink(temp_file)

    @pytest.mark.unit
    async def test_validate_invalid_yaml_file(
        self, validation_config: KubernetesValidationConfig
    ):
        """無効なYAMLファイルの検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        # 無効なYAMLファイルを作成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [\n")  # 構文エラー
            temp_file = f.name

        try:
            with pytest.raises(ManifestValidationError):
                await validator.validate_file(temp_file)

        finally:
            os.unlink(temp_file)

    @pytest.mark.unit
    async def test_validate_directory(
        self, validation_config: KubernetesValidationConfig,
        sample_deployment_manifest: dict[str, Any],
        sample_service_manifest: dict[str, Any]
    ):
        """ディレクトリの検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        # 一時ディレクトリに複数のマニフェストファイルを作成
        with tempfile.TemporaryDirectory() as temp_dir:
            # Deploymentファイル
            deployment_file = Path(temp_dir) / "deployment.yaml"
            with open(deployment_file, 'w') as f:
                yaml.dump(sample_deployment_manifest, f)

            # Serviceファイル
            service_file = Path(temp_dir) / "service.yaml"
            with open(service_file, 'w') as f:
                yaml.dump(sample_service_manifest, f)

            results = await validator.validate_directory(temp_dir)

            assert len(results) == 2
            assert all(result.is_valid for result in results)


class TestKubernetesValidatorIntegration:
    """Kubernetes検証統合テスト"""

    @pytest.mark.integration
    async def test_comprehensive_manifest_validation(
        self, validation_config: KubernetesValidationConfig, sample_deployment_manifest: dict[str, Any]
    ):
        """包括的なマニフェスト検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        # 全検証機能を実行
        results = await validator.validate_comprehensive(sample_deployment_manifest)

        assert len(results) >= 4  # 基本、セキュリティ、リソース、ヘルスチェック
        assert all(result.is_valid for result in results)

        # 各検証タイプが含まれることを確認
        validation_types = [result.validation_type for result in results]
        assert "syntax" in validation_types
        assert "security" in validation_types
        assert "resources" in validation_types
        assert "health_checks" in validation_types

    @pytest.mark.integration
    async def test_multi_manifest_validation_performance(
        self, validation_config: KubernetesValidationConfig,
        sample_deployment_manifest: dict[str, Any],
        sample_service_manifest: dict[str, Any],
        sample_configmap_manifest: dict[str, Any],
    ):
        """複数マニフェスト検証のパフォーマンステスト"""
        validator = KubernetesManifestValidator(config=validation_config)

        manifests = [
            sample_deployment_manifest,
            sample_service_manifest,
            sample_configmap_manifest,
        ] * 10  # 30個のマニフェスト

        import time
        start_time = time.time()

        # 並列検証実行
        all_results = await validator.validate_multiple_manifests(manifests)

        end_time = time.time()
        execution_time = end_time - start_time

        # パフォーマンス要件検証
        assert execution_time < 10.0  # 10秒以内
        assert len(all_results) == 30
        assert all(result.is_valid for result in all_results)

    @pytest.mark.integration
    async def test_validation_with_real_kubernetes_manifests(
        self, validation_config: KubernetesValidationConfig
    ):
        """実際のKubernetesマニフェストによる検証"""
        validator = KubernetesManifestValidator(config=validation_config)

        # 実際のプロダクション用マニフェスト例
        production_manifests = [
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": "spec-rag-api",
                    "namespace": "production",
                    "labels": {
                        "app": "spec-rag-api",
                        "version": "v1.0.0",
                        "environment": "production",
                    },
                    "annotations": {
                        "prometheus.io/scrape": "true",
                    },
                },
                "spec": {
                    "replicas": 5,
                    "selector": {"matchLabels": {"app": "spec-rag-api"}},
                    "template": {
                        "metadata": {"labels": {"app": "spec-rag-api"}},
                        "spec": {
                            "securityContext": {"runAsNonRoot": True},
                            "containers": [
                                {
                                    "name": "api",
                                    "image": "spec-rag-api:v1.0.0",
                                    "resources": {
                                        "requests": {"cpu": "1", "memory": "2Gi"},
                                        "limits": {"cpu": "2", "memory": "4Gi"},
                                    },
                                    "livenessProbe": {
                                        "httpGet": {"path": "/health", "port": 8000},
                                        "initialDelaySeconds": 30,
                                    },
                                    "readinessProbe": {
                                        "httpGet": {"path": "/ready", "port": 8000},
                                        "initialDelaySeconds": 10,
                                    },
                                    "securityContext": {
                                        "allowPrivilegeEscalation": False,
                                        "readOnlyRootFilesystem": True,
                                    },
                                }
                            ],
                        },
                    },
                },
            }
        ]

        for manifest in production_manifests:
            results = await validator.validate_comprehensive(manifest)

            # 本番環境レベルの品質を要求
            assert all(result.is_valid for result in results)
            assert all(result.severity in [ValidationSeverity.INFO, ValidationSeverity.WARNING] for result in results)
