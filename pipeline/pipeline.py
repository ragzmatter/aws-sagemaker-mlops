"""SageMaker Pipeline (v3 SDK): Preprocess -> Train -> Evaluate -> [AUC gate] -> Register.

CHANGED vs your pipeline.py (reasoning in REVIEW_NOTES.md)
  1. Removed the duplicated second copy of the script that was pasted after the
     first one (it sat inside a stray triple-quote string).
  2. Added EvaluateModel (ProcessingStep + PropertyFile) and a ConditionStep gate on
     AUC. Registration now happens only if the model clears the gate; the
     metrics file is attached to the model package (Model quality tab in the UI).
  3. BUG FIX: the registered model used the *training* image. Serving needs
     image_scope="inference".
  4. inference.py is bundled into model.tar.gz through ModelBuilder(source_code=...).
     Before, the registered package had no custom handler, so the real-time endpoint
     could not have served your 11-column CSV contract.
  5. Dropped the separate CreateModel step. Every run created an orphan
     "pipelines-...-CreateModel-..." model (you deleted several by hand in your
     history). The registry only needs the package. If register() ever complains
     that the model is not built, re-add ModelStep(step_args=model_builder.build()).
  6. Processing outputs go to a per-execution S3 prefix (Join + PIPELINE_EXECUTION_ID).
     The fixed s3://<bucket>/<project>/train prefix was overwritten every run,
     losing reproducibility and confusing step caching.
  7. Real hyperparameters; MinAuc is a pipeline parameter; pipeline gets tags.
  8. response_types is application/json (what inference.py returns), not text/csv.

NOT YET RUN against AWS - I could verify imports/signatures against the published
v3 packages, but not execute the pipeline. Expect to iterate once on first upsert.
"""
import builtins
import os

# ---- Windows line-ending workaround (kept from your version) -------------------
# ModelTrainer writes sm_train.sh with the platform default newline (CRLF on Windows),
# which breaks bash inside the Linux container. Force LF for text-mode writes.
_original_open = builtins.open


def _lf_open(file, mode="r", *args, **kwargs):
    if "w" in mode and "b" not in mode and "newline" not in kwargs:
        kwargs["newline"] = "\n"
    return _original_open(file, mode, *args, **kwargs)


builtins.open = _lf_open

from sagemaker.core import image_uris  # noqa: E402
from sagemaker.core.helper.session_helper import get_execution_role  # noqa: E402
from sagemaker.core.model_metrics import MetricsSource, ModelMetrics  # noqa: E402
from sagemaker.core.processing import ScriptProcessor  # noqa: E402
from sagemaker.core.shapes import (  # noqa: E402
    ProcessingInput, ProcessingOutput, ProcessingS3Input, ProcessingS3Output,
)
from sagemaker.core.training.configs import Compute, InputData, SourceCode  # noqa: E402  (sagemaker.train.configs is a deprecated shim)
from sagemaker.core.workflow.conditions import ConditionGreaterThanOrEqualTo  # noqa: E402
from sagemaker.core.workflow.execution_variables import ExecutionVariables  # noqa: E402
from sagemaker.core.workflow.functions import Join, JsonGet  # noqa: E402
from sagemaker.core.workflow.parameters import ParameterFloat, ParameterString  # noqa: E402
from sagemaker.core.workflow.pipeline_context import PipelineSession  # noqa: E402
from sagemaker.core.workflow.properties import PropertyFile  # noqa: E402
from sagemaker.mlops.workflow.condition_step import ConditionStep  # noqa: E402
from sagemaker.mlops.workflow.model_step import ModelStep  # noqa: E402
from sagemaker.mlops.workflow.pipeline import Pipeline  # noqa: E402
from sagemaker.mlops.workflow.steps import CacheConfig, ProcessingStep, TrainingStep  # noqa: E402
from sagemaker.serve.model_builder import ModelBuilder  # noqa: E402
from sagemaker.train import ModelTrainer  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
region = os.environ["AWS_REGION"]
bucket = os.environ["BUCKET"]
role = os.environ.get("ROLE_ARN") or get_execution_role()

PROJECT = "sagemaker-mlops-lab"
PIPELINE_NAME = os.environ.get("PIPELINE_NAME", "CustomerConversionPipeline")
MODEL_GROUP = os.environ.get("MODEL_GROUP", "CustomerConversionModelGroup")
SKLEARN_VERSION = "1.2-1"

pipeline_session = PipelineSession(default_bucket=bucket)
cache_config = CacheConfig(enable_caching=True, expire_after="P7D")  # ISO-8601 duration

# --------------------------------------------------------------------------
# Parameters (visible/overridable when you Start an execution in Studio)
# --------------------------------------------------------------------------
processing_instance_type = ParameterString("ProcessingInstanceType", default_value="ml.m5.large")
training_instance_type = ParameterString("TrainingInstanceType", default_value="ml.m5.large")
model_approval_status = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")
input_data = ParameterString("InputData", default_value=f"s3://{bucket}/raw/customers.csv")
min_auc = ParameterFloat("MinAuc", default_value=0.65)

# Images: same sklearn container, but the correct *scope* per use.
def _img(scope=None):
    kw = dict(framework="sklearn", region=region, version=SKLEARN_VERSION,
              py_version="py3", instance_type="ml.m5.large")
    if scope:
        kw["image_scope"] = scope
    return image_uris.retrieve(**kw)


processing_image = _img()
training_image = _img("training")
inference_image = _img("inference")

# Per-execution S3 locations
def _exec_prefix(name):
    return Join(on="/", values=["s3:/", bucket, PROJECT, "executions",
                                ExecutionVariables.PIPELINE_EXECUTION_ID, name])


# --------------------------------------------------------------------------
# 1. Preprocess
# --------------------------------------------------------------------------
processor = ScriptProcessor(
    image_uri=processing_image, command=["python3"], role=role,
    instance_type=processing_instance_type, instance_count=1,
    base_job_name="customer-preprocess", sagemaker_session=pipeline_session,
)

step_process = ProcessingStep(
    name="PreprocessData",
    step_args=processor.run(
        code="src/preprocess.py",
        inputs=[ProcessingInput(input_name="raw", s3_input=ProcessingS3Input(
            s3_uri=input_data, local_path="/opt/ml/processing/input",
            s3_data_type="S3Prefix", s3_input_mode="File"))],
        outputs=[
            ProcessingOutput(output_name="train", s3_output=ProcessingS3Output(
                s3_uri=_exec_prefix("train"), local_path="/opt/ml/processing/output/train",
                s3_upload_mode="EndOfJob")),
            ProcessingOutput(output_name="test", s3_output=ProcessingS3Output(
                s3_uri=_exec_prefix("test"), local_path="/opt/ml/processing/output/test",
                s3_upload_mode="EndOfJob")),
        ],
    ),
    cache_config=cache_config,
)

# --------------------------------------------------------------------------
# 2. Train
# --------------------------------------------------------------------------
model_trainer = ModelTrainer(
    training_image=training_image, role=role,
    source_code=SourceCode(source_dir="src", entry_script="train.py"),
    compute=Compute(instance_type=training_instance_type, instance_count=1),
    base_job_name="customer-training", sagemaker_session=pipeline_session,
    hyperparameters={"n-estimators": 200, "max-depth": 12, "min-samples-leaf": 20},
    input_data_config=[
        InputData(channel_name="train",
                  data_source=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri),
        InputData(channel_name="test",
                  data_source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri),
    ],
)

step_train = TrainingStep(name="TrainModel", step_args=model_trainer.train(), cache_config=cache_config)

# --------------------------------------------------------------------------
# 3. Evaluate
# --------------------------------------------------------------------------
evaluator = ScriptProcessor(
    image_uri=processing_image, command=["python3"], role=role,
    instance_type=processing_instance_type, instance_count=1,
    base_job_name="customer-evaluate", sagemaker_session=pipeline_session,
)

evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")

step_eval = ProcessingStep(
    name="EvaluateModel",
    step_args=evaluator.run(
        code="src/evaluate.py",
        inputs=[
            ProcessingInput(input_name="model", s3_input=ProcessingS3Input(
                s3_uri=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                local_path="/opt/ml/processing/model", s3_data_type="S3Prefix", s3_input_mode="File")),
            ProcessingInput(input_name="test", s3_input=ProcessingS3Input(
                s3_uri=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                local_path="/opt/ml/processing/test", s3_data_type="S3Prefix", s3_input_mode="File")),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", s3_output=ProcessingS3Output(
            s3_uri=_exec_prefix("evaluation"), local_path="/opt/ml/processing/evaluation",
            s3_upload_mode="EndOfJob"))],
    ),
    property_files=[evaluation_report],
    cache_config=cache_config,
)

# --------------------------------------------------------------------------
# 4. Register (inference image + bundled inference.py + metrics)
# --------------------------------------------------------------------------
model_metrics = ModelMetrics(model_statistics=MetricsSource(
    s3_uri=Join(on="/", values=[
        step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
        "evaluation.json"]),
    content_type="application/json",
))

model_builder = ModelBuilder(
    s3_model_data_url=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    image_uri=inference_image, role_arn=role, sagemaker_session=pipeline_session,
    source_code=SourceCode(source_dir="src", entry_script="inference.py"),
)

step_register = ModelStep(
    name="RegisterCustomerModel",
    step_args=model_builder.register(
        content_types=["text/csv", "application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_GROUP,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    ),
)

# --------------------------------------------------------------------------
# 5. Quality gate
# --------------------------------------------------------------------------
step_gate = ConditionStep(
    name="CheckAUC",
    conditions=[ConditionGreaterThanOrEqualTo(
        left=JsonGet(step_name=step_eval.name, property_file=evaluation_report,
                     json_path="binary_classification_metrics.auc.value"),
        right=min_auc)],
    if_steps=[step_register],
    else_steps=[],
)

pipeline = Pipeline(
    name=PIPELINE_NAME,
    parameters=[processing_instance_type, training_instance_type, model_approval_status, input_data, min_auc],
    steps=[step_process, step_train, step_eval, step_gate],
    sagemaker_session=pipeline_session,
)

if __name__ == "__main__":
    pipeline.upsert(role_arn=role, tags=[{"Key": "Project", "Value": PROJECT}])
    print("Pipeline created/updated:", pipeline.describe()["PipelineArn"])
