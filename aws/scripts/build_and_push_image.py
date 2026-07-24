"""Build the custom training image and push it to ECR.

SageMaker can only pull training images from ECR (AWS's private registry) —
Docker Hub is not an option — so the flow is: create the repo, authenticate the
local Docker daemon against it, build, tag, push. Idempotent: an existing repo
is reused, and pushing the same tag overwrites it.

Requires a RUNNING Docker daemon (Docker Desktop on Windows). This is the one
script in aws/ that needs local Docker; everything else is pure API calls.

    python aws/scripts/build_and_push_image.py --tag latest
"""

import argparse
import base64
import logging
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# boto3 imported lazily in main() so the pure helper stays CI-testable.

REPOSITORY = "demand-forecasting-training"
DOCKERFILE_DIR = Path(__file__).resolve().parent.parent / "docker"


def image_uri(account: str, region: str, repository: str, tag: str) -> str:
    """The fully-qualified ECR image URI. Pure — unit-tested in CI."""
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{repository}:{tag}"


def ensure_repository(ecr: Any, repository: str) -> None:
    try:
        ecr.create_repository(
            repositoryName=repository,
            imageScanningConfiguration={"scanOnPush": True},
            tags=[{"Key": "project", "Value": "demand-forecasting"}],
        )
        logger.info("Created ECR repository %s", repository)
    except ecr.exceptions.RepositoryAlreadyExistsException:
        logger.info("ECR repository %s already exists", repository)


def docker_login(ecr: Any, registry: str) -> None:
    """Exchange AWS credentials for a Docker registry login (token lasts 12h)."""
    token = ecr.get_authorization_token()["authorizationData"][0]["authorizationToken"]
    user, password = base64.b64decode(token).decode().split(":", 1)
    subprocess.run(
        ["docker", "login", "--username", user, "--password-stdin", registry],
        input=password.encode(), check=True,
    )
    logger.info("Authenticated Docker against %s", registry)


def build_and_push(uri: str, dockerfile: str = "Dockerfile") -> None:
    # --platform linux/amd64 is explicit: SageMaker runs amd64, and an image
    # silently built for another arch fails at job start, not at build time.
    subprocess.run(
        [
            "docker", "build", "--platform", "linux/amd64",
            "-f", str(DOCKERFILE_DIR / dockerfile), "-t", uri, str(DOCKERFILE_DIR),
        ],
        check=True,
    )
    subprocess.run(["docker", "push", uri], check=True)
    logger.info("Pushed %s", uri)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument(
        "--dockerfile", default="Dockerfile",
        help="File within aws/docker/, e.g. Dockerfile.gpu for the TFT image",
    )
    args = parser.parse_args()

    import boto3

    session = boto3.Session()
    account = session.client("sts").get_caller_identity()["Account"]
    region = session.region_name
    ecr = session.client("ecr")

    uri = image_uri(account, region, args.repository, args.tag)
    ensure_repository(ecr, args.repository)
    docker_login(ecr, uri.split("/")[0])
    build_and_push(uri, args.dockerfile)
    logger.info("Train with: python aws/scripts/run_sagemaker_training.py --image-uri %s ...", uri)


if __name__ == "__main__":
    main()
