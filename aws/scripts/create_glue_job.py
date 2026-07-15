"""Create (or update) and optionally run the Glue feature job.

Uploads aws/glue/feature_job.py to s3://<bucket>/glue-scripts/, registers a Glue
job pointing at it, and (with --run) starts a run. Idempotent: re-running updates
the existing job definition rather than failing.

Prerequisite (one-time, Console): a Glue **service role** that can read/write the
bucket — its ARN is passed as --role-arn. See aws/README / PROGRESS for the steps.

    python aws/scripts/create_glue_job.py --bucket <name> --role-arn <arn> --run --wait
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import boto3

logger = logging.getLogger(__name__)

JOB_NAME = "demand-forecasting-features"
SCRIPT = Path(__file__).resolve().parent.parent / "glue" / "feature_job.py"


def upload_script(s3: Any, bucket: str) -> str:
    key = "glue-scripts/feature_job.py"
    s3.upload_file(str(SCRIPT), bucket, key)
    uri = f"s3://{bucket}/{key}"
    logger.info("Uploaded Glue script to %s", uri)
    return uri


def create_or_update_job(glue: Any, script_uri: str, role_arn: str, bucket: str) -> None:
    command = {"Name": "glueetl", "ScriptLocation": script_uri, "PythonVersion": "3"}
    # --bucket is our job arg; the others are Glue conventions/observability.
    default_args = {"--bucket": bucket, "--job-language": "python", "--enable-metrics": "true"}
    settings = dict(
        Role=role_arn,
        Command=command,
        DefaultArguments=default_args,
        GlueVersion="4.0",
        WorkerType="G.1X",
        NumberOfWorkers=10,
    )
    try:
        glue.create_job(Name=JOB_NAME, **settings)
        logger.info("Created Glue job %s", JOB_NAME)
    except glue.exceptions.AlreadyExistsException:
        glue.update_job(JobName=JOB_NAME, JobUpdate=settings)
        logger.info("Updated existing Glue job %s", JOB_NAME)


def run_job(glue: Any, wait: bool) -> str:
    run_id = glue.start_job_run(JobName=JOB_NAME)["JobRunId"]
    logger.info("Started run %s", run_id)
    if not wait:
        return run_id
    while True:
        state = glue.get_job_run(JobName=JOB_NAME, RunId=run_id)["JobRun"]["JobRunState"]
        logger.info("Run %s: %s", run_id, state)
        if state in ("SUCCEEDED", "FAILED", "STOPPED", "TIMEOUT"):
            return run_id
        time.sleep(30)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="S3 bucket")
    parser.add_argument("--role-arn", required=True, help="Glue service role ARN")
    parser.add_argument("--run", action="store_true", help="Start a run after registering")
    parser.add_argument("--wait", action="store_true", help="Poll until the run finishes")
    args = parser.parse_args()

    s3 = boto3.client("s3")
    glue = boto3.client("glue")
    script_uri = upload_script(s3, args.bucket)
    create_or_update_job(glue, script_uri, args.role_arn, args.bucket)
    if args.run:
        run_job(glue, args.wait)


if __name__ == "__main__":
    main()
