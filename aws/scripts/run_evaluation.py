"""Launch the evaluation panel as a SageMaker Processing job.

A Processing job is the generic sibling of a training job: same ephemeral
lifecycle (provision -> pull image -> download inputs -> run -> upload outputs
-> destroy), but no toolkit and no env-var contract — inputs land at the
/opt/ml/processing/... paths this launcher chooses, and only the declared
output directories are uploaded. Evaluation produces metric tables, not a
model, so this is the right job type; Phase 6's pipeline and Phase 7's Model
Monitor reuse the same primitive.

Code delivery is explicit here (training's sourcedir.tar.gz magic does not
exist for plain Processors): `stage_code` uploads the entry scripts, the
portable `demand_forecasting` package, and config.yaml to S3 as one prefix;
that prefix is mounted as a ProcessingInput and PYTHONPATH points at it.
Reuses the Phase-3 ECR image unchanged — it already has every dependency.

    python aws/scripts/run_evaluation.py --bucket <name> --role-arn <sagemaker-role-arn>
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# boto3/sagemaker are imported lazily inside main() (repo convention) so the
# pure helpers below stay CI-testable without the optional `aws` group.

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CODE_MOUNT = "/opt/ml/processing/input/code"
FEATURES_MOUNT = "/opt/ml/processing/input/features"
OUTPUT_MOUNT = "/opt/ml/processing/output/evaluation"


def code_files(repo_root: Path) -> list[tuple[Path, str]]:
    """(local path, container-relative key) pairs for the staged code tree.

    The layout must let `evaluate_entry.py` do `from train_entry import ...` and
    `from demand_forecasting... import ...` with PYTHONPATH set to the code
    root: entry scripts + config.yaml at the root, the package as a subtree.
    Pure — unit-tested in CI.
    """
    pairs: list[tuple[Path, str]] = []
    for script in sorted((repo_root / "aws" / "sagemaker").glob("*.py")):
        pairs.append((script, script.name))
    pairs.append((repo_root / "config" / "config.yaml", "config.yaml"))
    package_root = repo_root / "src"
    for module in sorted((package_root / "demand_forecasting").rglob("*.py")):
        pairs.append((module, module.relative_to(package_root).as_posix()))
    return pairs


def stage_code(s3: Any, bucket: str, run_name: str) -> str:
    """Upload the code tree to S3; returns the prefix URI to mount."""
    prefix = f"processing-code/{run_name}"
    for local, key in code_files(REPO_ROOT):
        s3.upload_file(str(local), bucket, f"{prefix}/{key}")
    uri = f"s3://{bucket}/{prefix}/"
    logger.info("Staged %d code files to %s", len(code_files(REPO_ROOT)), uri)
    return uri


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument(
        "--image-uri", help="ECR image; defaults to this account's Phase-3 :latest"
    )
    parser.add_argument(
        "--max-series", type=int, default=3000,
        help="Same bounded default as training, for an apples-to-apples panel",
    )
    args = parser.parse_args()

    import boto3
    import sagemaker_compat
    from build_and_push_image import REPOSITORY, image_uri
    from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor

    sagemaker_compat.apply()  # Windows SDK infinite-loop fix; see sagemaker_compat.py

    session = boto3.Session()
    run_name = f"eval-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}"
    uri = args.image_uri or image_uri(
        session.client("sts").get_caller_identity()["Account"],
        session.region_name,
        REPOSITORY,
        "latest",
    )
    code_uri = stage_code(session.client("s3"), args.bucket, run_name)

    processor = Processor(
        role=args.role_arn,
        image_uri=uri,
        instance_count=1,
        instance_type=args.instance_type,
        entrypoint=["python3", f"{CODE_MOUNT}/evaluate_entry.py"],
        env={"PYTHONPATH": CODE_MOUNT, "PYTHONUNBUFFERED": "1"},
        base_job_name="demand-forecasting-eval",
        tags=[{"Key": "project", "Value": "demand-forecasting"}],
    )
    processor.run(
        inputs=[
            ProcessingInput(source=code_uri, destination=CODE_MOUNT, input_name="code"),
            ProcessingInput(
                source=f"s3://{args.bucket}/features/",
                destination=FEATURES_MOUNT,
                input_name="features",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source=OUTPUT_MOUNT,
                destination=f"s3://{args.bucket}/evaluation/{run_name}/",
                output_name="evaluation",
            )
        ],
        arguments=[
            "--features-dir", FEATURES_MOUNT,
            "--output-dir", OUTPUT_MOUNT,
            "--config", f"{CODE_MOUNT}/config.yaml",
            "--max-series", str(args.max_series),
        ],
    )
    logger.info("Evaluation artifacts: s3://%s/evaluation/%s/", args.bucket, run_name)


if __name__ == "__main__":
    main()
