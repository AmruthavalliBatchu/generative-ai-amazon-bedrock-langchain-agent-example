#!/bin/bash
cd .. 
cd agent/lambda/agent-handler
# Name of the zip file
ZIP_FILE="agent_deployment_package.zip"

# S3 Bucket and key/path
# OLD_S3_BUCKET="avivabedrockapp-174671970284"
S3_BUCKET="avivabedrockapp-719514420056"
S3_KEY="agent/lambda/agent-handler/${ZIP_FILE}"
S3_URI="s3://${S3_BUCKET}/${S3_KEY}"

# Lambda ARN
# OLD_LAMBDA_ARN="arn:aws:lambda:us-east-1:174671970284:function:avivabedrockapp-GenAILexHandler"
LAMBDA_ARN="arn:aws:lambda:us-east-1:719514420056:function:avivabedrockapp-GenAILexHandler"

# Step 1: Zip all Python files in the current directory
echo "Zipping Python files into ${ZIP_FILE}..."
zip -r ${ZIP_FILE} *.py

# Step 2: Upload to S3
echo "Uploading ${ZIP_FILE} to ${S3_URI}..."
aws s3 cp ${ZIP_FILE} ${S3_URI}

# Step 3: Update Lambda function
echo "Updating Lambda function ${LAMBDA_ARN} with the new deployment package from S3..."
aws lambda update-function-code \
    --function-name ${LAMBDA_ARN} \
    --s3-bucket ${S3_BUCKET} \
    --s3-key ${S3_KEY}

echo "Update complete!"

