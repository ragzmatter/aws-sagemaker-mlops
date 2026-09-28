<#
 Rebuild the lab foundation in ~2 minutes. Idempotent: safe to re-run.
   .\scripts\bootstrap.ps1                # role + bucket + data upload
   .\scripts\bootstrap.ps1 -WithStudio    # also domain + user profile (needed to VIEW pipelines in the Studio UI)
 Run from the repo root. Dot-source to keep the env vars in your shell:  . .\scripts\bootstrap.ps1
#>
param([switch]$WithStudio)

$env:AWS_REGION  = "ap-south-1"
$env:PROJECT     = "sagemaker-mlops-lab"
$env:ROLE_NAME   = "SageMaker-MLOps-Lab-ExecutionRole"
$env:PIPELINE_NAME = "CustomerConversionPipeline"
$env:MODEL_GROUP   = "CustomerConversionModelGroup"
$env:ENDPOINT_NAME = "customer-conversion-endpoint"
$env:ACCOUNT_ID  = aws sts get-caller-identity --query Account --output text
if (-not $env:ACCOUNT_ID) { throw "AWS credentials not working (aws sts get-caller-identity failed)." }
$env:BUCKET      = "sagemaker-mlops-lab-$($env:ACCOUNT_ID)"   # 'sagemaker' in the name lets AmazonSageMakerFullAccess reach it

# ---- IAM execution role ----------------------------------------------------
aws iam get-role --role-name $env:ROLE_NAME 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
  @'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}
'@ | Out-File -Encoding ascii trust-policy.json
  aws iam create-role --role-name $env:ROLE_NAME --assume-role-policy-document file://trust-policy.json --tags Key=Project,Value=$env:PROJECT | Out-Null
  aws iam attach-role-policy --role-name $env:ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
  Start-Sleep -Seconds 10   # IAM is eventually consistent
}
$env:ROLE_ARN = aws iam get-role --role-name $env:ROLE_NAME --query Role.Arn --output text

# ---- S3 bucket -------------------------------------------------------------
aws s3api head-bucket --bucket $env:BUCKET 2>$null
if ($LASTEXITCODE -ne 0) {
  aws s3api create-bucket --bucket $env:BUCKET --region $env:AWS_REGION `
    --create-bucket-configuration LocationConstraint=$env:AWS_REGION | Out-Null
  aws s3api put-bucket-tagging --bucket $env:BUCKET --tagging "TagSet=[{Key=Project,Value=$($env:PROJECT)}]"
}

# ---- Data ------------------------------------------------------------------
if (-not (Test-Path data\output\customers.csv)) { python data\generate_data.py }
aws s3 cp data\output\customers.csv "s3://$($env:BUCKET)/raw/customers.csv"

# ---- Studio (optional) -----------------------------------------------------
if ($WithStudio) {
  $env:VPC_ID = aws ec2 describe-vpcs --region $env:AWS_REGION --filters Name=isDefault,Values=true --query "Vpcs[0].VpcId" --output text
  $subnets = (aws ec2 describe-subnets --region $env:AWS_REGION --filters Name=vpc-id,Values=$env:VPC_ID --query "Subnets[*].SubnetId" --output text) -split "\s+"
  $env:DOMAIN_ID = aws sagemaker list-domains --region $env:AWS_REGION --query "Domains[?DomainName=='sagemaker-mlops-domain'].DomainId | [0]" --output text
  if (-not $env:DOMAIN_ID -or $env:DOMAIN_ID -eq "None") {
    $env:DOMAIN_ID = aws sagemaker create-domain --region $env:AWS_REGION --domain-name sagemaker-mlops-domain `
      --auth-mode IAM --vpc-id $env:VPC_ID --subnet-ids $subnets `
      --default-user-settings "ExecutionRole=$($env:ROLE_ARN)" `
      --default-space-settings "ExecutionRole=$($env:ROLE_ARN)" `
      --tags Key=Project,Value=$env:PROJECT --query DomainArn --output text
    $env:DOMAIN_ID = ($env:DOMAIN_ID -split "/")[-1]
  }
  do { Start-Sleep 15; $st = aws sagemaker describe-domain --domain-id $env:DOMAIN_ID --region $env:AWS_REGION --query Status --output text; "Domain: $st" } while ($st -eq "Pending")
  aws sagemaker create-user-profile --region $env:AWS_REGION --domain-id $env:DOMAIN_ID --user-profile-name raghav `
    --user-settings "ExecutionRole=$($env:ROLE_ARN)" --tags Key=Project,Value=$env:PROJECT Key=Owner,Value=Raghav 2>$null | Out-Null
}

"`nREADY  region=$($env:AWS_REGION)  bucket=$($env:BUCKET)`nROLE_ARN=$($env:ROLE_ARN)"
