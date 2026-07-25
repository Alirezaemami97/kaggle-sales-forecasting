"""The durable infrastructure as CDK — the AWS-native IaC twin of what the boto3
setup scripts and the console built by hand: the data-lake bucket, the SageMaker
and Glue roles, the two training ECR repos, the forecast-WAPE alarm, and the
(disabled) schedule + retrain EventBridge rules, spanning five service types.

Synth-only: we do NOT deploy over the working hand-built resources. This is the
reviewed, reproducible-from-code proof — the CDK analogue of `terraform plan`,
matching the portable track's "validate/plan only" stance.

The SageMaker Pipeline is intentionally absent: it is already code (`pipeline.py`,
upserted by `run_pipeline.py`); re-expressing it as a CfnPipeline would duplicate
it. This stack owns the durable infra; the SDK owns the pipeline definition.
"""

from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    Tags,
)
from aws_cdk import (
    aws_cloudwatch as cloudwatch,
)
from aws_cdk import (
    aws_ecr as ecr,
)
from aws_cdk import (
    aws_events as events,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_s3 as s3,
)
from constructs import Construct

PIPELINE_NAME = "demand-forecasting-ct"
ALARM_NAME = "demand-forecasting-forecast-wape"
ECR_REPOS = ("demand-forecasting-training", "demand-forecasting-training-gpu")


class InfraStack(Stack):
    def __init__(self, scope: Construct, cid: str, *, bucket_name: str, **kwargs: object) -> None:
        super().__init__(scope, cid, **kwargs)
        Tags.of(self).add("project", "demand-forecasting")

        # Data lake. RETAIN so `cdk destroy` never deletes the forecasts/models.
        bucket = s3.Bucket(
            self,
            "DataLake",
            bucket_name=bucket_name,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # Execution roles — SageMaker jobs and the Glue job assume these.
        sagemaker_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            role_name="demand-forecasting-sagemaker-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
            ],
        )
        bucket.grant_read_write(sagemaker_role)

        glue_role = iam.Role(
            self,
            "GlueJobRole",
            role_name="demand-forecasting-glue-role",
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSGlueServiceRole")
            ],
        )
        bucket.grant_read_write(glue_role)

        # Training images. RETAIN so destroy never orphans in-use images.
        for repo in ECR_REPOS:
            ecr.Repository(
                self,
                repo.replace("-", " ").title().replace(" ", ""),
                repository_name=repo,
                removal_policy=RemovalPolicy.RETAIN,
            )

        # The Phase-7 alarm the retrain rule watches.
        cloudwatch.Alarm(
            self,
            "ForecastWapeAlarm",
            alarm_name=ALARM_NAME,
            metric=cloudwatch.Metric(
                namespace="DemandForecasting",
                metric_name="ForecastWAPE",
                period=Duration.minutes(5),
                statistic="Average",
            ),
            threshold=0.75,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        # The events→pipeline trust the boto3 launchers needed but couldn't
        # self-grant (the Phase 6/7 gotcha) — codified here, least-privilege.
        pipeline_arn = self.format_arn(
            service="sagemaker", resource="pipeline", resource_name=PIPELINE_NAME
        )
        events_role = iam.Role(
            self, "EventsToPipelineRole", assumed_by=iam.ServicePrincipal("events.amazonaws.com")
        )
        events_role.add_to_policy(
            iam.PolicyStatement(
                actions=["sagemaker:StartPipelineExecution"], resources=[pipeline_arn]
            )
        )
        target = events.CfnRule.TargetProperty(
            id="ct-pipeline", arn=pipeline_arn, role_arn=events_role.role_arn
        )

        # Both rules DISABLED — teardown discipline in code (Phase 6/7).
        events.CfnRule(
            self,
            "WeeklyScheduleRule",
            name="demand-forecasting-ct-weekly",
            schedule_expression="rate(7 days)",
            state="DISABLED",
            targets=[target],
        )
        events.CfnRule(
            self,
            "RetrainRule",
            name="demand-forecasting-ct-retrain",
            state="DISABLED",
            event_pattern={
                "source": ["aws.cloudwatch"],
                "detail-type": ["CloudWatch Alarm State Change"],
                "detail": {
                    "alarmName": [ALARM_NAME],
                    "state": {"value": ["ALARM"]},
                },
            },
            targets=[target],
        )
