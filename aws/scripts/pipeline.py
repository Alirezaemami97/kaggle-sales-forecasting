"""Phase 6: the continuous-training pipeline as a SageMaker Pipelines DAG.

    Train ─┐
           ├─▶ CheckWAPE (wape ≤ threshold?) ──true──▶ Register (Pending)
Evaluate ──┘                                 └─false──▶ (skip)

Defined in Python, compiled to JSON, and executed by SageMaker — not this laptop.
That is the whole point: the launchers in this folder orchestrate from your
machine and stall if it sleeps; a Pipeline hands the DAG to SageMaker, which runs
the steps, passes artifacts through S3, and applies the gate on its own. This is
the AWS-native form of the local M7 continuous-training loop.

It reuses the exact pieces the standalone launchers built — `build_estimator`
(training), the Phase-4a eval Processor + `evaluate_entry`, the `register_model`
image/group — so a scheduled run and a hand-run cannot drift.

Honest scoping: `evaluate_entry` runs a rolling-origin BACKTEST (it retrains per
fold), so the gate's WAPE measures how this CONFIGURATION generalises; the
TrainingStep produces the deployable artifact (same config, trained on all
origins). Gating the artifact on the backtest is the standard "backtest to
decide, train-on-all to deploy" pattern — not a claim that the exact artifact was
scored. Data prep (the Glue job) is upstream and separately scheduled, so the
pipeline reads the existing `features/` as its input rather than re-running Glue.
"""

import logging
from typing import Any

from register_model import MODEL_PACKAGE_GROUP
from run_evaluation import CODE_MOUNT, FEATURES_MOUNT, OUTPUT_MOUNT
from run_sagemaker_training import build_estimator

logger = logging.getLogger(__name__)

PIPELINE_NAME = "demand-forecasting-ct"
# The default gate. Loose on purpose: the baseline backtest WAPE is ~0.70, so
# this passes a healthy model and blocks a broken one, without hand-tuning.
DEFAULT_WAPE_THRESHOLD = 0.75
EVAL_OUTPUT_URI_SUFFIX = "pipeline-evaluation"


def get_pipeline(
    *,
    role_arn: str,
    bucket: str,
    image_uri: str,
    code_uri: str,
    session: Any,
    group: str = MODEL_PACKAGE_GROUP,
) -> Any:
    """Build the Pipeline object. `session` is a PipelineSession (so .fit/.run/
    .register return deferred step args instead of executing); `code_uri` is the
    eval code bundle already staged to S3 by run_pipeline.py."""
    from sagemaker.inputs import TrainingInput
    from sagemaker.model import Model
    from sagemaker.model_metrics import MetricsSource, ModelMetrics
    from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
    from sagemaker.workflow.execution_variables import ExecutionVariables
    from sagemaker.workflow.functions import Join, JsonGet
    from sagemaker.workflow.model_step import ModelStep
    from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.steps import ProcessingStep, TrainingStep

    # Parameters are resolved at execution time, so one definition runs many ways
    # (a bigger sample, a stricter gate) without re-compiling.
    max_series = ParameterInteger(name="MaxSeries", default_value=3000)
    wape_threshold = ParameterFloat(name="WapeThreshold", default_value=DEFAULT_WAPE_THRESHOLD)
    approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    train_instance = ParameterString(name="TrainInstanceType", default_value="ml.m5.xlarge")
    eval_instance = ParameterString(name="EvalInstanceType", default_value="ml.m5.xlarge")

    features_uri = f"s3://{bucket}/features/"
    # Execution-scoped so each registered version pins its OWN metrics: Phase 8's
    # promote-if-beats reads these back, and a shared path would make every version
    # resolve to the latest eval, not its own snapshot.
    eval_base = f"s3://{bucket}/{EVAL_OUTPUT_URI_SUFFIX}"
    eval_output_uri = Join(on="/", values=[eval_base, ExecutionVariables.PIPELINE_EXECUTION_ID])
    eval_metrics_uri = Join(on="/", values=[eval_output_uri, "evaluation.json"])

    # 1. Train — produces the deployable artifact (same Estimator as the launcher).
    estimator = build_estimator(
        image_uri, role_arn, train_instance, bucket, {"max-series": max_series}, session
    )
    train_step = TrainingStep(
        name="Train", step_args=estimator.fit(inputs={"features": TrainingInput(features_uri)})
    )

    # 2. Evaluate — rolling-origin backtest of the same config (Phase-4a Processor).
    processor = Processor(
        role=role_arn,
        image_uri=image_uri,
        instance_count=1,
        instance_type=eval_instance,
        entrypoint=["python3", f"{CODE_MOUNT}/evaluate_entry.py"],
        env={"PYTHONPATH": CODE_MOUNT, "PYTHONUNBUFFERED": "1"},
        base_job_name="demand-forecasting-ct-eval",
        sagemaker_session=session,
        tags=[{"Key": "project", "Value": "demand-forecasting"}],
    )
    eval_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    eval_step = ProcessingStep(
        name="Evaluate",
        step_args=processor.run(
            inputs=[
                ProcessingInput(source=code_uri, destination=CODE_MOUNT, input_name="code"),
                ProcessingInput(
                    source=features_uri, destination=FEATURES_MOUNT, input_name="features"
                ),
            ],
            outputs=[
                ProcessingOutput(
                    source=OUTPUT_MOUNT, destination=eval_output_uri, output_name="evaluation"
                )
            ],
            arguments=[
                "--features-dir", FEATURES_MOUNT,
                "--output-dir", OUTPUT_MOUNT,
                "--config", f"{CODE_MOUNT}/config.yaml",
                "--max-series", max_series.to_string(),
            ],
        ),
        property_files=[eval_report],
    )

    # 3. Register — reached only on the gate's true branch. The backtest WAPE is
    # attached as ModelMetrics, so a later incumbent-lookup (Phase 8) can read it.
    model = Model(
        image_uri=image_uri,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=role_arn,
        sagemaker_session=session,
    )
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(s3_uri=eval_metrics_uri, content_type="application/json")
    )
    register_step = ModelStep(
        name="Register",
        step_args=model.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=group,
            approval_status=approval_status,
            model_metrics=model_metrics,
        ),
    )

    # 4. The gate — a printed metric can't branch a DAG, only a PropertyFile a
    # downstream step reads with JsonGet. Register only if WAPE clears the bar.
    gate = ConditionStep(
        name="CheckWAPE",
        conditions=[
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=eval_report,
                    json_path="wape_item_store",
                ),
                right=wape_threshold,
            )
        ],
        if_steps=[register_step],
        else_steps=[],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[max_series, wape_threshold, approval_status, train_instance, eval_instance],
        steps=[train_step, eval_step, gate],
        sagemaker_session=session,
    )
