"""CI/CD as code: a CodePipeline that pulls from GitHub and runs the same
ruff/mypy/pytest gate in CodeBuild that GitHub Actions runs — so the checks are
identical whether CI is GitHub-native or AWS-native. Synth-only here; wiring the
GitHub source needs a one-time CodeStar connection whose ARN comes from context
(`cdk.json`). A cdk-deploy stage is the documented extension once the connection
is authorized.
"""

from aws_cdk import (
    Stack,
)
from aws_cdk import (
    aws_codebuild as codebuild,
)
from aws_cdk import (
    aws_codepipeline as codepipeline,
)
from aws_cdk import (
    aws_codepipeline_actions as cpactions,
)
from constructs import Construct

_PLACEHOLDER_CONNECTION = (
    "arn:aws:codestar-connections:us-east-1:000000000000:connection/PLACEHOLDER"
)


class PipelineStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kwargs: object) -> None:
        super().__init__(scope, cid, **kwargs)
        connection_arn = self.node.try_get_context("connection_arn") or _PLACEHOLDER_CONNECTION
        owner = self.node.try_get_context("github_owner") or "OWNER"
        repo = self.node.try_get_context("github_repo") or "REPO"

        source_output = codepipeline.Artifact()
        source_action = cpactions.CodeStarConnectionsSourceAction(
            action_name="GitHub",
            owner=owner,
            repo=repo,
            branch="main",
            connection_arn=connection_arn,
            output=source_output,
        )
        gate_project = codebuild.PipelineProject(
            self,
            "Gate",
            build_spec=codebuild.BuildSpec.from_source_filename("buildspec.yml"),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.STANDARD_7_0
            ),
        )
        gate_action = cpactions.CodeBuildAction(
            action_name="Gate", project=gate_project, input=source_output
        )

        codepipeline.Pipeline(
            self,
            "CtCicd",
            pipeline_name="demand-forecasting-cicd",
            stages=[
                codepipeline.StageProps(stage_name="Source", actions=[source_action]),
                codepipeline.StageProps(stage_name="Gate", actions=[gate_action]),
            ],
        )
