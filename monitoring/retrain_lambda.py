"""Lambda: start the training pipeline when the drift alarm fires.

CHANGED: skips if an execution is already Executing (an alarm can flap and would
otherwise queue duplicate, billable retrains). The new model still lands in the
registry as PendingManualApproval - retraining is automatic, promotion is not.
"""
import os

import boto3

sm = boto3.client("sagemaker")
PIPELINE = os.environ["PIPELINE_NAME"]


def lambda_handler(event, context):
    running = sm.list_pipeline_executions(PipelineName=PIPELINE, SortBy="CreationTime",
                                          SortOrder="Descending", MaxResults=5)
    if any(e["PipelineExecutionStatus"] == "Executing" for e in running["PipelineExecutionSummaries"]):
        print("Execution already running; skipping.")
        return {"started": False}
    arn = sm.start_pipeline_execution(PipelineName=PIPELINE,
                                      PipelineExecutionDescription="Auto-retrain: feature drift alarm")["PipelineExecutionArn"]
    print("Retraining started:", arn)
    return {"started": True, "pipelineExecutionArn": arn}
