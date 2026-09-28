<#
 Tear the lab down in dependency order and verify. Safe to re-run; missing resources are skipped.
   .\scripts\teardown.ps1                 # everything billable + bucket, KEEPS the IAM role
   .\scripts\teardown.ps1 -KeepBucket -KeepStudio
   .\scripts\teardown.ps1 -DeleteRole
 Ends with a verification report - read it. 'Deleted' is not the same as 'gone from billing'.
#>
param([switch]$KeepBucket, [switch]$KeepStudio, [switch]$DeleteRole)
$R = "ap-south-1"
$EP = "customer-conversion-endpoint"; $PL = "CustomerConversionPipeline"; $MG = "CustomerConversionModelGroup"
$ACCOUNT = aws sts get-caller-identity --query Account --output text
$BUCKET = "sagemaker-mlops-lab-$ACCOUNT"
function Q { param([scriptblock]$s) try { & $s 2>$null | Out-Null } catch {} }

# 1 Automation (drift -> retrain)
Q { aws events remove-targets --rule sagemaker-mlops-drift-retraining --ids 1 --region $R }
Q { aws events delete-rule --name sagemaker-mlops-drift-retraining --region $R }
Q { aws lambda delete-function --function-name sagemaker-mlops-retrain --region $R }
Q { aws cloudwatch delete-alarms --alarm-names sagemaker-customer-conversion-5xx customer-conversion-feature-drift --region $R }

# 2 Serving: endpoint -> configs -> models  (deleting an endpoint does NOT delete its config/model)
Q { aws sagemaker delete-endpoint --endpoint-name $EP --region $R }
foreach ($c in (aws sagemaker list-endpoint-configs --region $R --name-contains customer-conversion --query "EndpointConfigs[].EndpointConfigName" --output text) -split "\s+") { if ($c) { Q { aws sagemaker delete-endpoint-config --endpoint-config-name $c --region $R } } }
foreach ($m in (aws sagemaker list-models --region $R --query "Models[?contains(ModelName,'customer-conversion') || contains(ModelName,'CreateModel')].ModelName" --output text) -split "\s+") { if ($m) { Q { aws sagemaker delete-model --model-name $m --region $R } } }

# 3 Running jobs
foreach ($j in (aws sagemaker list-training-jobs --status-equals InProgress --region $R --query "TrainingJobSummaries[].TrainingJobName" --output text) -split "\s+") { if ($j) { Q { aws sagemaker stop-training-job --training-job-name $j --region $R } } }
foreach ($j in (aws sagemaker list-processing-jobs --status-equals InProgress --region $R --query "ProcessingJobSummaries[].ProcessingJobName" --output text) -split "\s+") { if ($j) { Q { aws sagemaker stop-processing-job --processing-job-name $j --region $R } } }

# 4 Pipeline, then registry (versions first, then the group)
Q { aws sagemaker delete-pipeline --pipeline-name $PL --region $R }
foreach ($a in (aws sagemaker list-model-packages --model-package-group-name $MG --region $R --query "ModelPackageSummaryList[].ModelPackageArn" --output text) -split "\s+") { if ($a) { Q { aws sagemaker delete-model-package --model-package-name $a --region $R } } }
Q { aws sagemaker delete-model-package-group --model-package-group-name $MG --region $R }

# 5 Studio: apps -> spaces -> user profiles -> domain (+EFS)
if (-not $KeepStudio) {
  foreach ($d in (aws sagemaker list-domains --region $R --query "Domains[].DomainId" --output text) -split "\s+") { if (-not $d) { continue }
    foreach ($line in (aws sagemaker list-apps --domain-id-equals $d --region $R --query "Apps[?Status!='Deleted'].[AppType,AppName,UserProfileName,SpaceName]" --output text)) {
      if (-not $line) { continue }; $t,$n,$u,$s = $line -split "\t"
      if ($u -and $u -ne "None") { Q { aws sagemaker delete-app --domain-id $d --app-type $t --app-name $n --user-profile-name $u --region $R } }
      else                       { Q { aws sagemaker delete-app --domain-id $d --app-type $t --app-name $n --space-name $s --region $R } } }
    Start-Sleep 30
    foreach ($s in (aws sagemaker list-spaces --domain-id-equals $d --region $R --query "Spaces[].SpaceName" --output text) -split "\s+") { if ($s) { Q { aws sagemaker delete-space --domain-id $d --space-name $s --region $R } } }
    foreach ($u in (aws sagemaker list-user-profiles --domain-id-equals $d --region $R --query "UserProfiles[].UserProfileName" --output text) -split "\s+") { if ($u) { Q { aws sagemaker delete-user-profile --domain-id $d --user-profile-name $u --region $R } } }
    Start-Sleep 20
    Q { aws sagemaker delete-domain --domain-id $d --retention-policy HomeEfsFileSystem=Delete --region $R }
  }
}

# 6 Storage
if (-not $KeepBucket) { Q { aws s3 rm "s3://$BUCKET" --recursive }; Q { aws s3api delete-bucket --bucket $BUCKET --region $R } }
if ($DeleteRole) { Q { aws iam detach-role-policy --role-name SageMaker-MLOps-Lab-ExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess }; Q { aws iam delete-role --role-name SageMaker-MLOps-Lab-ExecutionRole } }

# ---- Verification ----------------------------------------------------------
"`n=== VERIFY (all should be empty) ==="
"Endpoints:        " + (aws sagemaker list-endpoints --region $R --query "Endpoints[].EndpointName" --output text)
"Endpoint configs: " + (aws sagemaker list-endpoint-configs --region $R --query "EndpointConfigs[].EndpointConfigName" --output text)
"Models:           " + (aws sagemaker list-models --region $R --query "Models[].ModelName" --output text)
"Pipelines:        " + (aws sagemaker list-pipelines --region $R --query "PipelineSummaries[].PipelineName" --output text)
"Domains:          " + (aws sagemaker list-domains --region $R --query "Domains[].DomainId" --output text)
"EFS file systems: " + (aws efs describe-file-systems --region $R --query "FileSystems[].FileSystemId" --output text)
"Buckets (lab):    " + (aws s3api list-buckets --query "Buckets[?starts_with(Name,'sagemaker-mlops-lab')].Name" --output text)
"Running jobs:     " + (aws sagemaker list-training-jobs --status-equals InProgress --region $R --query "TrainingJobSummaries[].TrainingJobName" --output text)
"`nAlso glance at: Billing -> Cost Explorer (group by Service) tomorrow; charges post with a delay."
