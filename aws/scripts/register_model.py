"""Register a trained artifact in the SageMaker Model Registry.

The Registry is the AWS analogue of the MLflow registry used locally: a Model
Package Group is the model's identity (`demand-forecaster-lgbm`), and each
training run adds a versioned Model Package under it, carrying an approval
status. That status — PendingManualApproval -> Approved — is AWS's answer to
MLflow's stages, and it is the switch Phase 8's promote/rollback will flip. We
register as PENDING on purpose: a model becomes production because it beat the
incumbent on WAPE, never because it finished training.

CreateModelPackage requires an InferenceSpecification (a container image + the
artifact) even with no endpoint in sight, so we name our own training image as a
placeholder. It has the deps to load the boosters but no serving code yet — real
inference lands in Phase 5 (Batch Transform). Honest scaffolding, not a working
serving path.

    python aws/scripts/register_model.py --role-arn <sagemaker-role-arn> \
        --model-data s3://<bucket>/models/<job>/output/model.tar.gz
"""

import argparse
import logging
from typing import Any

from aws_errors import already_exists

logger = logging.getLogger(__name__)

# boto3 imported lazily in main() so pure helpers stay CI-testable.

MODEL_PACKAGE_GROUP = "demand-forecaster-lgbm"


def ensure_group(sm: Any, group: str) -> None:
    """Create the Model Package Group if absent — idempotent, like every other
    setup script in aws/ (re-running must never be destructive)."""
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=group,
            ModelPackageGroupDescription=(
                "Quantile LightGBM demand forecaster (one booster per quantile), "
                "trained by the SageMaker script-mode job in aws/sagemaker/train_entry.py"
            ),
            Tags=[{"Key": "project", "Value": "demand-forecasting"}],
        )
        logger.info("Created model package group %s", group)
    except sm.exceptions.ClientError as exc:
        # CreateModelPackageGroup declares no typed "already exists" error, so a
        # duplicate must be recognised from the error shape — see aws_errors.
        if not already_exists(exc):
            raise
        logger.info("Model package group %s already exists", group)


def register(sm: Any, group: str, image_uri: str, model_data: str) -> str:
    resp = sm.create_model_package(
        ModelPackageGroupName=group,
        ModelPackageDescription="LightGBM quantile baseline from SageMaker script-mode training",
        InferenceSpecification={
            "Containers": [{"Image": image_uri, "ModelDataUrl": model_data}],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        # Approval is earned by evaluation (Phase 4/6), never granted at training time.
        ModelApprovalStatus="PendingManualApproval",
    )
    arn: str = resp["ModelPackageArn"]
    return arn


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data", required=True, help="S3 URI of model.tar.gz")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--group", default=MODEL_PACKAGE_GROUP)
    parser.add_argument(
        "--image-uri", help="ECR image for the inference spec; defaults to this account's :latest"
    )
    args = parser.parse_args()

    import boto3
    import sagemaker_compat
    from build_and_push_image import REPOSITORY, image_uri

    sagemaker_compat.apply()  # Windows SDK infinite-loop fix; see sagemaker_compat.py

    boto_session = boto3.Session()
    sm = boto_session.client("sagemaker")
    uri = args.image_uri or image_uri(
        boto_session.client("sts").get_caller_identity()["Account"],
        boto_session.region_name,
        REPOSITORY,
        "latest",
    )

    ensure_group(sm, args.group)
    arn = register(sm, args.group, uri, args.model_data)
    logger.info("Registered model package: %s", arn)
    logger.info(
        "Status is PendingManualApproval. Approve in the console "
        "(SageMaker > Model registry > %s) or via UpdateModelPackage once it has "
        "earned it on the evaluation panel.", args.group
    )


if __name__ == "__main__":
    main()
