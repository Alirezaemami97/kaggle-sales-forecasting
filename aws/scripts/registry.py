"""Model Registry helpers for the Phase 8 deployment harness.

The SageMaker registry has no named stages: a version carries a
`ModelApprovalStatus` (PendingManualApproval -> Approved / Rejected) and the
convention is "the latest Approved version is production". Promote = Approve the
candidate; rollback = Reject the current top so the prior Approved is latest
again. A batch job reads latest-Approved at run time, so rollback is one API
call with no redeploy.
"""

import json
import logging
from typing import Any

from register_model import MODEL_PACKAGE_GROUP

logger = logging.getLogger(__name__)

APPROVED = "Approved"
PENDING = "PendingManualApproval"
REJECTED = "Rejected"


def list_versions(sm: Any, group: str = MODEL_PACKAGE_GROUP) -> list[dict[str, Any]]:
    """All versions of the group, newest first."""
    resp = sm.list_model_packages(
        ModelPackageGroupName=group,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=100,
    )
    return resp.get("ModelPackageSummaryList", [])


def latest_with_status(versions: list[dict[str, Any]], status: str) -> dict[str, Any] | None:
    """The newest version with the given approval status (incumbent = Approved,
    candidate = PendingManualApproval). Pure — unit-tested in CI."""
    for pkg in versions:
        if pkg.get("ModelApprovalStatus") == status:
            return pkg
    return None


def metric_wape(sm: Any, s3: Any, package_arn: str) -> float | None:
    """The backtest WAPE attached as ModelMetrics at registration, or None if the
    version carries no metrics (e.g. the Phase-3 v1 registered before the pipeline)."""
    desc = sm.describe_model_package(ModelPackageName=package_arn)
    uri = (
        desc.get("ModelMetrics", {})
        .get("ModelStatistics", {})
        .get("Statistics", {})
        .get("S3Uri")
    )
    if not uri:
        return None
    bucket, key = uri.replace("s3://", "", 1).split("/", 1)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return float(json.loads(body)["wape_item_store"])


def set_approval(sm: Any, package_arn: str, status: str) -> None:
    """Flip a version's approval status — the promote/rollback primitive."""
    sm.update_model_package(ModelPackageArn=package_arn, ModelApprovalStatus=status)
    logger.info("Set %s -> %s", package_arn, status)
