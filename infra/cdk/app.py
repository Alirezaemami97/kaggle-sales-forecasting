#!/usr/bin/env python3
"""CDK app entry point. `cdk synth` (or `python app.py`) compiles both stacks to
CloudFormation without touching AWS — the offline, free IaC proof. Deploy is
deliberately not part of the workflow (the hand-built infra already runs); this
is synth/diff only, the CDK analogue of the portable track's `terraform plan`.
"""

import aws_cdk as cdk

from demand_forecasting_cdk.infra_stack import InfraStack
from demand_forecasting_cdk.pipeline_stack import PipelineStack

app = cdk.App()
bucket_name = app.node.try_get_context("bucket_name") or "demand-forecasting-bucket"

InfraStack(app, "DemandForecastingInfra", bucket_name=bucket_name)
PipelineStack(app, "DemandForecastingCicd")

app.synth()
