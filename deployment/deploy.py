"""Deploy the newest APPROVED model package to a real-time endpoint with Data Capture.

Plain boto3 on purpose: each call maps 1:1 to the CLI commands you already know
(create-model / create-endpoint-config / create-endpoint / update-endpoint), and it
is independent of which SageMaker Python SDK major version is installed.

  python deployment/deploy.py                       # newest Approved package
  python deployment/deploy.py --model-package-arn arn:...
  python deployment/deploy.py --canary --alarm sagemaker-customer-conversion-5xx   # on updates

Console: SageMaker AI -> Inference -> Models / Endpoint configurations / Endpoints.
"""
import argparse
import os
import time

import boto3

REGION = os.environ.get("AWS_REGION", "ap-south-1")
sm = boto3.client("sagemaker", region_name=REGION)


def latest_approved(group):
    r = sm.list_model_packages(ModelPackageGroupName=group, ModelApprovalStatus="Approved",
                               SortBy="CreationTime", SortOrder="Descending", MaxResults=1)
    if not r["ModelPackageSummaryList"]:
        raise SystemExit(f"No Approved package in {group}. Approve one first (Studio -> Models, or update-model-package).")
    return r["ModelPackageSummaryList"][0]["ModelPackageArn"]


def endpoint_exists(name):
    try:
        sm.describe_endpoint(EndpointName=name)
        return True
    except sm.exceptions.ClientError as e:
        if "Could not find" in str(e):
            return False
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", default=os.environ.get("ENDPOINT_NAME", "customer-conversion-endpoint"))
    p.add_argument("--model-group", default=os.environ.get("MODEL_GROUP", "CustomerConversionModelGroup"))
    p.add_argument("--model-package-arn")
    p.add_argument("--instance-type", default="ml.m5.large")
    p.add_argument("--instance-count", type=int, default=1)
    p.add_argument("--sampling", type=int, default=100, help="Data Capture sampling %%")
    p.add_argument("--canary", action="store_true", help="on UPDATE: shift traffic in a canary step")
    p.add_argument("--alarm", action="append", default=[], help="CloudWatch alarm(s) that trigger auto-rollback")
    a = p.parse_args()

    role, bucket = os.environ["ROLE_ARN"], os.environ["BUCKET"]
    arn = a.model_package_arn or latest_approved(a.model_group)
    version = arn.rsplit("/", 1)[-1]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"{a.endpoint_name}-v{version}-{stamp}"
    config_name = f"{a.endpoint_name}-cfg-v{version}-{stamp}"

    print("Model package:", arn)
    sm.create_model(ModelName=model_name, ExecutionRoleArn=role,
                    Containers=[{"ModelPackageName": arn}])

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic", "ModelName": model_name,
            "InstanceType": a.instance_type, "InitialInstanceCount": a.instance_count,
            "InitialVariantWeight": 1.0}],
        DataCaptureConfig={
            "EnableCapture": True, "InitialSamplingPercentage": a.sampling,
            "DestinationS3Uri": f"s3://{bucket}/monitoring/data-capture",
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
            "CaptureContentTypeHeader": {"CsvContentTypes": ["text/csv"], "JsonContentTypes": ["application/json"]}},
        Tags=[{"Key": "Project", "Value": "sagemaker-mlops-lab"}],
    )

    if endpoint_exists(a.endpoint_name):
        kwargs = {}
        if a.canary:
            kwargs["DeploymentConfig"] = {
                "BlueGreenUpdatePolicy": {
                    "TrafficRoutingConfiguration": {
                        "Type": "CANARY",
                        "CanarySize": {"Type": "CAPACITY_PERCENT", "Value": 50},
                        "WaitIntervalInSeconds": 300},
                    "TerminationWaitInSeconds": 120,
                    "MaximumExecutionTimeoutInSeconds": 1800},
                **({"AutoRollbackConfiguration": {"Alarms": [{"AlarmName": x} for x in a.alarm]}} if a.alarm else {})}
        sm.update_endpoint(EndpointName=a.endpoint_name, EndpointConfigName=config_name, **kwargs)
        print("Updating endpoint", a.endpoint_name, "(canary)" if a.canary else "")
    else:
        sm.create_endpoint(EndpointName=a.endpoint_name, EndpointConfigName=config_name,
                           Tags=[{"Key": "Project", "Value": "sagemaker-mlops-lab"}])
        print("Creating endpoint", a.endpoint_name)

    sm.get_waiter("endpoint_in_service").wait(EndpointName=a.endpoint_name,
                                              WaiterConfig={"Delay": 30, "MaxAttempts": 60})
    print("InService:", a.endpoint_name)


if __name__ == "__main__":
    main()
