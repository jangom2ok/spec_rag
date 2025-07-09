"""Kubernetesマニフェスト検証サービス

TDD実装：マニフェストのテスト/検証
- マニフェスト構文検証: YAML形式、必須フィールド、リソース制限
- セキュリティ検証: セキュリティコンテキスト、RBAC、ネットワークポリシー
- 設定検証: ConfigMap、Secret、環境変数の整合性
- ヘルスチェック検証: Liveness/Readiness Probe設定
- リソース検証: CPU/メモリ制限、ストレージ要件
"""

import asyncio
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """検証結果の重要度"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ResourceUnit(str, Enum):
    """リソース単位"""

    CPU_MILLICORES = "m"
    MEMORY_BYTES = "B"
    MEMORY_KILOBYTES = "K"
    MEMORY_MEGABYTES = "M"
    MEMORY_GIGABYTES = "G"
    MEMORY_KIBIBYTES = "Ki"
    MEMORY_MEBIBYTES = "Mi"
    MEMORY_GIBIBYTES = "Gi"


@dataclass
class KubernetesValidationConfig:
    """Kubernetes検証設定"""

    enable_security_validation: bool = True
    enable_resource_validation: bool = True
    enable_health_check_validation: bool = True
    strict_mode: bool = False

    # ラベル・アノテーション要件
    required_labels: list[str] = field(default_factory=lambda: ["app", "version"])
    required_annotations: list[str] = field(default_factory=list)

    # リソース制限
    max_cpu_limit: str = "2"
    max_memory_limit: str = "4Gi"
    min_replicas: int = 1
    max_replicas: int = 10

    # セキュリティ設定
    allow_privileged_containers: bool = False
    require_non_root: bool = True
    require_read_only_root_filesystem: bool = True

    # ヘルスチェック設定
    require_liveness_probe: bool = True
    require_readiness_probe: bool = True
    min_initial_delay_seconds: int = 5
    max_timeout_seconds: int = 10

    def __post_init__(self):
        """設定値のバリデーション"""
        if self.min_replicas <= 0:
            raise ValueError("min_replicas must be greater than 0")
        if self.max_replicas <= self.min_replicas:
            raise ValueError("max_replicas must be greater than min_replicas")


@dataclass
class ValidationResult:
    """検証結果"""

    is_valid: bool
    severity: ValidationSeverity
    message: str
    validation_type: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    validated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


@dataclass
class SecurityValidation:
    """セキュリティ検証結果"""

    is_valid: bool
    has_security_context: bool = False
    runs_as_non_root: bool = False
    read_only_root_filesystem: bool = False
    has_privileged_containers: bool = False
    allows_privilege_escalation: bool = False
    security_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


@dataclass
class ResourceRequirements:
    """リソース要件検証結果"""

    is_valid: bool
    has_resource_limits: bool = False
    has_resource_requests: bool = False
    cpu_limit: str = ""
    memory_limit: str = ""
    cpu_request: str = ""
    memory_request: str = ""
    resource_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


@dataclass
class HealthCheckValidation:
    """ヘルスチェック検証結果"""

    is_valid: bool
    has_liveness_probe: bool = False
    has_readiness_probe: bool = False
    liveness_config: dict[str, Any] = field(default_factory=dict)
    readiness_config: dict[str, Any] = field(default_factory=dict)
    health_check_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


@dataclass
class LabelAnnotationValidation:
    """ラベル・アノテーション検証結果"""

    is_valid: bool
    has_required_labels: bool = False
    has_required_annotations: bool = False
    missing_labels: list[str] = field(default_factory=list)
    missing_annotations: list[str] = field(default_factory=list)
    label_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)


@dataclass
class FileValidationResult:
    """ファイル検証結果"""

    file_path: str
    is_valid: bool
    validation_results: list[ValidationResult]
    parse_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result["validation_results"] = [vr.to_dict() for vr in self.validation_results]
        return result


class ManifestValidationError(Exception):
    """マニフェスト検証エラー"""

    pass


class KubernetesManifestValidator:
    """Kubernetesマニフェスト検証サービスメインクラス"""

    def __init__(self, config: KubernetesValidationConfig):
        self.config = config

        # 検証統計
        self._validation_count = 0
        self._error_count = 0
        self._warning_count = 0

        # キャッシュ
        self._validation_cache: dict[str, ValidationResult] = {}

    async def validate_manifest(self, manifest: dict[str, Any]) -> ValidationResult:
        """マニフェストの基本検証"""
        self._validation_count += 1
        errors: list[str] = []
        warnings: list[str] = []

        try:
            # 必須フィールドの検証
            if not self._validate_required_fields(manifest, errors):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Missing required fields",
                    validation_type="syntax",
                    errors=errors,
                )

            # APIバージョンの検証
            if not self._validate_api_version(manifest, errors):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Invalid API version",
                    validation_type="syntax",
                    errors=errors,
                )

            # Kindの検証
            if not self._validate_kind(manifest, errors, warnings):
                if errors:
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message="Invalid Kind",
                        validation_type="syntax",
                        errors=errors,
                    )

            # メタデータの検証
            self._validate_metadata(manifest, errors, warnings)

            # Specの検証
            self._validate_spec(manifest, errors, warnings)

            is_valid = len(errors) == 0
            severity = (
                ValidationSeverity.ERROR
                if errors
                else (
                    ValidationSeverity.WARNING if warnings else ValidationSeverity.INFO
                )
            )

            if errors:
                self._error_count += 1
            elif warnings:
                self._warning_count += 1

            return ValidationResult(
                is_valid=is_valid,
                severity=severity,
                message=(
                    f"Valid {manifest.get('kind', 'Unknown')} manifest"
                    if is_valid
                    else "Validation failed"
                ),
                validation_type="syntax",
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            logger.error(f"Validation error: {e}")
            self._error_count += 1
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation exception: {str(e)}",
                validation_type="syntax",
                errors=[str(e)],
            )

    def _validate_required_fields(
        self, manifest: dict[str, Any], errors: list[str]
    ) -> bool:
        """必須フィールドの検証"""
        required_fields = ["apiVersion", "kind", "metadata"]

        for required_field in required_fields:
            if required_field not in manifest:
                errors.append(f"Missing required field: {required_field}")
                return False

        return True

    def _validate_api_version(
        self, manifest: dict[str, Any], errors: list[str]
    ) -> bool:
        """APIバージョンの検証"""
        api_version = manifest.get("apiVersion", "")

        # 有効なAPIバージョンパターン
        valid_patterns = [
            r"^v1$",
            r"^apps/v1$",
            r"^extensions/v1beta1$",
            r"^networking\.k8s\.io/v1$",
            r"^rbac\.authorization\.k8s\.io/v1$",
            r"^policy/v1beta1$",
        ]

        if not any(re.match(pattern, api_version) for pattern in valid_patterns):
            errors.append(f"Invalid apiVersion: {api_version}")
            return False

        return True

    def _validate_kind(
        self, manifest: dict[str, Any], errors: list[str], warnings: list[str]
    ) -> bool:
        """Kindの検証"""
        kind = manifest.get("kind", "")

        # サポートされているKind
        supported_kinds = [
            "Deployment",
            "Service",
            "ConfigMap",
            "Secret",
            "Ingress",
            "PersistentVolume",
            "PersistentVolumeClaim",
            "ServiceAccount",
            "Role",
            "RoleBinding",
            "ClusterRole",
            "ClusterRoleBinding",
            "NetworkPolicy",
            "Pod",
            "ReplicaSet",
            "DaemonSet",
            "StatefulSet",
        ]

        if not kind:
            errors.append("Missing or empty kind field")
            return False

        if kind not in supported_kinds:
            warnings.append(f"Unsupported or deprecated kind: {kind}")

        return True

    def _validate_metadata(
        self, manifest: dict[str, Any], errors: list[str], warnings: list[str]
    ) -> None:
        """メタデータの検証"""
        metadata = manifest.get("metadata", {})

        # 名前の検証
        name = metadata.get("name", "")
        if not name:
            errors.append("Missing metadata.name")
        elif not re.match(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", name):
            errors.append(f"Invalid metadata.name format: {name}")

        # ネームスペースの検証
        namespace = metadata.get("namespace", "")
        if namespace and not re.match(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$", namespace):
            errors.append(f"Invalid namespace format: {namespace}")

    def _validate_spec(
        self, manifest: dict[str, Any], errors: list[str], warnings: list[str]
    ) -> None:
        """Specの検証"""
        kind = manifest.get("kind", "")
        spec = manifest.get("spec", {})

        if kind in ["Deployment", "ReplicaSet", "DaemonSet", "StatefulSet"]:
            self._validate_workload_spec(spec, errors, warnings)
        elif kind == "Service":
            self._validate_service_spec(spec, errors, warnings)

    def _validate_workload_spec(
        self, spec: dict[str, Any], errors: list[str], warnings: list[str]
    ) -> None:
        """ワークロードSpecの検証"""
        # レプリカ数の検証
        replicas = spec.get("replicas", 1)
        if replicas < self.config.min_replicas:
            warnings.append(
                f"Replicas ({replicas}) below minimum ({self.config.min_replicas})"
            )
        elif replicas > self.config.max_replicas:
            warnings.append(
                f"Replicas ({replicas}) above maximum ({self.config.max_replicas})"
            )

        # セレクタの検証
        if "selector" not in spec:
            errors.append("Missing spec.selector")

    def _validate_service_spec(
        self, spec: dict[str, Any], errors: list[str], warnings: list[str]
    ) -> None:
        """ServiceSpecの検証"""
        # ポートの検証
        ports = spec.get("ports", [])
        if not ports:
            errors.append("Service must have at least one port")

        # セレクタの検証
        if "selector" not in spec:
            warnings.append("Service without selector (headless service)")

    async def validate_security(self, manifest: dict[str, Any]) -> SecurityValidation:
        """セキュリティ検証"""
        if not self.config.enable_security_validation:
            return SecurityValidation(is_valid=True)

        violations = []

        # Podテンプレートの取得
        pod_spec = self._extract_pod_spec(manifest)
        if not pod_spec:
            return SecurityValidation(is_valid=True)  # Pod仕様がない場合はスキップ

        # セキュリティコンテキストの検証
        has_security_context = "securityContext" in pod_spec
        runs_as_non_root = False
        read_only_root_filesystem = False
        has_privileged_containers = False
        allows_privilege_escalation = False

        if has_security_context:
            security_context = pod_spec["securityContext"]
            runs_as_non_root = security_context.get("runAsNonRoot", False)

            if not runs_as_non_root and self.config.require_non_root:
                violations.append("Container should run as non-root user")

        # コンテナセキュリティの検証
        containers = pod_spec.get("containers", [])
        for container in containers:
            container_security = container.get("securityContext", {})

            # 特権コンテナの検出
            if container_security.get("privileged", False):
                has_privileged_containers = True
                if not self.config.allow_privileged_containers:
                    violations.append("Privileged containers detected")

            # 権限昇格の検証
            if container_security.get("allowPrivilegeEscalation", True):
                allows_privilege_escalation = True
                violations.append("Privilege escalation allowed")

            # 読み取り専用ルートファイルシステムの検証
            if container_security.get("readOnlyRootFilesystem", False):
                read_only_root_filesystem = True
            elif self.config.require_read_only_root_filesystem:
                violations.append("Root filesystem should be read-only")

        is_valid = len(violations) == 0

        return SecurityValidation(
            is_valid=is_valid,
            has_security_context=has_security_context,
            runs_as_non_root=runs_as_non_root,
            read_only_root_filesystem=read_only_root_filesystem,
            has_privileged_containers=has_privileged_containers,
            allows_privilege_escalation=allows_privilege_escalation,
            security_violations=violations,
        )

    async def validate_resources(
        self, manifest: dict[str, Any]
    ) -> ResourceRequirements:
        """リソース検証"""
        if not self.config.enable_resource_validation:
            return ResourceRequirements(is_valid=True)

        violations = []

        # Podテンプレートの取得
        pod_spec = self._extract_pod_spec(manifest)
        if not pod_spec:
            return ResourceRequirements(is_valid=True)

        has_resource_limits = False
        has_resource_requests = False
        cpu_limit = ""
        memory_limit = ""
        cpu_request = ""
        memory_request = ""

        # コンテナリソースの検証
        containers = pod_spec.get("containers", [])
        for container in containers:
            resources = container.get("resources", {})

            # リソース制限の検証
            limits = resources.get("limits", {})
            if limits:
                has_resource_limits = True
                cpu_limit = limits.get("cpu", "")
                memory_limit = limits.get("memory", "")

                # CPU制限の検証
                if cpu_limit and not self._is_cpu_within_limit(cpu_limit):
                    violations.append(
                        f"CPU limit ({cpu_limit}) exceeds maximum ({self.config.max_cpu_limit})"
                    )

                # メモリ制限の検証
                if memory_limit and not self._is_memory_within_limit(memory_limit):
                    violations.append(
                        f"Memory limit ({memory_limit}) exceeds maximum ({self.config.max_memory_limit})"
                    )
            else:
                violations.append("Missing resource limits")

            # リソース要求の検証
            requests = resources.get("requests", {})
            if requests:
                has_resource_requests = True
                cpu_request = requests.get("cpu", "")
                memory_request = requests.get("memory", "")
            else:
                violations.append("Missing resource requests")

        is_valid = len(violations) == 0

        return ResourceRequirements(
            is_valid=is_valid,
            has_resource_limits=has_resource_limits,
            has_resource_requests=has_resource_requests,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
            cpu_request=cpu_request,
            memory_request=memory_request,
            resource_violations=violations,
        )

    def _is_cpu_within_limit(self, cpu_value: str) -> bool:
        """CPU制限値の検証"""
        try:
            # CPU値を数値に変換（ミリコア単位）
            if cpu_value.endswith("m"):
                cpu_millicores = int(cpu_value[:-1])
            else:
                cpu_millicores = int(float(cpu_value) * 1000)

            # 制限値をミリコア単位に変換
            if self.config.max_cpu_limit.endswith("m"):
                max_millicores = int(self.config.max_cpu_limit[:-1])
            else:
                max_millicores = int(float(self.config.max_cpu_limit) * 1000)

            return cpu_millicores <= max_millicores
        except (ValueError, TypeError):
            return False

    def _is_memory_within_limit(self, memory_value: str) -> bool:
        """メモリ制限値の検証"""
        try:
            memory_bytes = self._parse_memory_value(memory_value)
            max_memory_bytes = self._parse_memory_value(self.config.max_memory_limit)
            return memory_bytes <= max_memory_bytes
        except (ValueError, TypeError):
            return False

    def _parse_memory_value(self, memory_str: str) -> int:
        """メモリ値をバイト単位に変換"""
        if not memory_str:
            return 0

        # 単位マッピング
        units = {
            "B": 1,
            "K": 1000,
            "M": 1000**2,
            "G": 1000**3,
            "Ki": 1024,
            "Mi": 1024**2,
            "Gi": 1024**3,
        }

        # 数値と単位を分離
        for unit in sorted(units.keys(), key=len, reverse=True):
            if memory_str.endswith(unit):
                value = float(memory_str[: -len(unit)])
                return int(value * units[unit])

        # 単位なしの場合はバイト単位として扱う
        return int(float(memory_str))

    async def validate_health_checks(
        self, manifest: dict[str, Any]
    ) -> HealthCheckValidation:
        """ヘルスチェック検証"""
        if not self.config.enable_health_check_validation:
            return HealthCheckValidation(is_valid=True)

        violations: list[str] = []

        # Podテンプレートの取得
        pod_spec = self._extract_pod_spec(manifest)
        if not pod_spec:
            return HealthCheckValidation(is_valid=True)

        has_liveness_probe = False
        has_readiness_probe = False
        liveness_config = {}
        readiness_config = {}

        # コンテナヘルスチェックの検証
        containers = pod_spec.get("containers", [])
        for container in containers:
            # Liveness Probeの検証
            liveness_probe = container.get("livenessProbe")
            if liveness_probe:
                has_liveness_probe = True
                liveness_config = liveness_probe
                self._validate_probe_config(liveness_probe, "liveness", violations)
            elif self.config.require_liveness_probe:
                violations.append("Missing liveness probe")

            # Readiness Probeの検証
            readiness_probe = container.get("readinessProbe")
            if readiness_probe:
                has_readiness_probe = True
                readiness_config = readiness_probe
                self._validate_probe_config(readiness_probe, "readiness", violations)
            elif self.config.require_readiness_probe:
                violations.append("Missing readiness probe")

        is_valid = len(violations) == 0

        return HealthCheckValidation(
            is_valid=is_valid,
            has_liveness_probe=has_liveness_probe,
            has_readiness_probe=has_readiness_probe,
            liveness_config=liveness_config,
            readiness_config=readiness_config,
            health_check_violations=violations,
        )

    def _validate_probe_config(
        self, probe: dict[str, Any], probe_type: str, violations: list[str]
    ) -> None:
        """プローブ設定の検証"""
        # 初期遅延時間の検証
        initial_delay = probe.get("initialDelaySeconds", 0)
        if initial_delay < self.config.min_initial_delay_seconds:
            violations.append(f"{probe_type} probe initialDelaySeconds too short")

        # タイムアウト時間の検証
        timeout = probe.get("timeoutSeconds", 1)
        if timeout > self.config.max_timeout_seconds:
            violations.append(f"{probe_type} probe timeoutSeconds too long")

        # 期間の検証
        period = probe.get("periodSeconds", 10)
        if period < 1:
            violations.append(f"{probe_type} probe periodSeconds too short")

    async def validate_labels_and_annotations(
        self, manifest: dict[str, Any]
    ) -> LabelAnnotationValidation:
        """ラベル・アノテーション検証"""
        violations = []
        metadata = manifest.get("metadata", {})

        # ラベルの検証
        labels = metadata.get("labels", {})
        missing_labels = []

        for required_label in self.config.required_labels:
            if required_label not in labels:
                missing_labels.append(required_label)

        if missing_labels:
            violations.append(f"Missing required labels: {', '.join(missing_labels)}")

        # ラベル値の検証
        for label_key, label_value in labels.items():
            if not self._is_valid_label_value(label_value):
                violations.append(f"Invalid label format: {label_key}={label_value}")

        # アノテーションの検証
        annotations = metadata.get("annotations", {})
        missing_annotations = []

        for required_annotation in self.config.required_annotations:
            if required_annotation not in annotations:
                missing_annotations.append(required_annotation)

        if missing_annotations:
            violations.append(
                f"Missing required annotations: {', '.join(missing_annotations)}"
            )

        has_required_labels = len(missing_labels) == 0
        has_required_annotations = len(missing_annotations) == 0
        is_valid = len(violations) == 0

        return LabelAnnotationValidation(
            is_valid=is_valid,
            has_required_labels=has_required_labels,
            has_required_annotations=has_required_annotations,
            missing_labels=missing_labels,
            missing_annotations=missing_annotations,
            label_violations=violations,
        )

    def _is_valid_label_value(self, value: str) -> bool:
        """ラベル値の形式検証"""
        if not value:  # 空の値は無効
            return False

        # Kubernetesラベル値の形式規則
        return (
            re.match(r"^[a-zA-Z0-9]([-a-zA-Z0-9_.]*[a-zA-Z0-9])?$", value) is not None
        )

    def _extract_pod_spec(self, manifest: dict[str, Any]) -> dict[str, Any] | None:
        """マニフェストからPod仕様を抽出"""
        kind = manifest.get("kind", "")

        if kind == "Pod":
            return manifest.get("spec", {})
        elif kind in ["Deployment", "ReplicaSet", "DaemonSet", "StatefulSet"]:
            return manifest.get("spec", {}).get("template", {}).get("spec", {})

        return None

    async def validate_file(self, file_path: str) -> FileValidationResult:
        """YAMLファイルの検証"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # YAML解析
            try:
                documents = list(yaml.safe_load_all(content))
            except yaml.YAMLError as e:
                raise ManifestValidationError(f"YAML parsing error: {e}") from e

            # 各ドキュメントを検証
            validation_results = []
            for doc in documents:
                if doc:  # 空のドキュメントをスキップ
                    result = await self.validate_manifest(doc)
                    validation_results.append(result)

            is_valid = all(result.is_valid for result in validation_results)

            return FileValidationResult(
                file_path=file_path,
                is_valid=is_valid,
                validation_results=validation_results,
            )

        except FileNotFoundError as e:
            raise ManifestValidationError(f"File not found: {file_path}") from e
        except Exception as e:
            raise ManifestValidationError(f"File validation error: {e}") from e

    async def validate_directory(
        self, directory_path: str
    ) -> list[FileValidationResult]:
        """ディレクトリ内のYAMLファイルを検証"""
        directory = Path(directory_path)
        if not directory.exists():
            raise ManifestValidationError(f"Directory not found: {directory_path}")

        yaml_files = list(directory.glob("*.yaml")) + list(directory.glob("*.yml"))

        results = []
        for yaml_file in yaml_files:
            try:
                result = await self.validate_file(str(yaml_file))
                results.append(result)
            except ManifestValidationError as e:
                results.append(
                    FileValidationResult(
                        file_path=str(yaml_file),
                        is_valid=False,
                        validation_results=[],
                        parse_errors=[str(e)],
                    )
                )

        return results

    async def validate_comprehensive(
        self, manifest: dict[str, Any]
    ) -> list[ValidationResult]:
        """包括的なマニフェスト検証"""
        results = []

        # 基本検証
        syntax_result = await self.validate_manifest(manifest)
        results.append(syntax_result)

        # セキュリティ検証
        if self.config.enable_security_validation:
            security_validation = await self.validate_security(manifest)
            security_result = ValidationResult(
                is_valid=security_validation.is_valid,
                severity=(
                    ValidationSeverity.ERROR
                    if not security_validation.is_valid
                    else ValidationSeverity.INFO
                ),
                message="Security validation completed",
                validation_type="security",
                errors=security_validation.security_violations,
            )
            results.append(security_result)

        # リソース検証
        if self.config.enable_resource_validation:
            resource_validation = await self.validate_resources(manifest)
            resource_result = ValidationResult(
                is_valid=resource_validation.is_valid,
                severity=(
                    ValidationSeverity.ERROR
                    if not resource_validation.is_valid
                    else ValidationSeverity.INFO
                ),
                message="Resource validation completed",
                validation_type="resources",
                errors=resource_validation.resource_violations,
            )
            results.append(resource_result)

        # ヘルスチェック検証
        if self.config.enable_health_check_validation:
            health_validation = await self.validate_health_checks(manifest)
            health_result = ValidationResult(
                is_valid=health_validation.is_valid,
                severity=(
                    ValidationSeverity.ERROR
                    if not health_validation.is_valid
                    else ValidationSeverity.INFO
                ),
                message="Health check validation completed",
                validation_type="health_checks",
                errors=health_validation.health_check_violations,
            )
            results.append(health_result)

        # ラベル・アノテーション検証
        label_validation = await self.validate_labels_and_annotations(manifest)
        label_result = ValidationResult(
            is_valid=label_validation.is_valid,
            severity=(
                ValidationSeverity.WARNING
                if not label_validation.is_valid
                else ValidationSeverity.INFO
            ),
            message="Label and annotation validation completed",
            validation_type="labels_annotations",
            errors=label_validation.label_violations,
        )
        results.append(label_result)

        return results

    async def validate_multiple_manifests(
        self, manifests: list[dict[str, Any]]
    ) -> list[ValidationResult]:
        """複数マニフェストの並列検証"""
        tasks = [self.validate_manifest(manifest) for manifest in manifests]
        results = await asyncio.gather(*tasks)
        return results

    def get_validation_statistics(self) -> dict[str, Any]:
        """検証統計情報の取得"""
        return {
            "total_validations": self._validation_count,
            "total_errors": self._error_count,
            "total_warnings": self._warning_count,
            "success_rate": (self._validation_count - self._error_count)
            / max(self._validation_count, 1),
            "cache_size": len(self._validation_cache),
        }
