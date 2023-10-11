rm -rf bedrock-langchain.zip
# rm -rf python
# mkdir python
# cd python
# pip install --target . -r ../requirements.txt
# cd .. 
zip -r bedrock-langchain.zip python/
aws s3 cp bedrock-langchain.zip s3://${S3_ARTIFACT_BUCKET_NAME}/agent/lambda-layers/bedrock-langchain.zip
export BEDROCK_LANGCHAIN_LAYER_ARN=$(aws lambda publish-layer-version \
    --layer-name bedrock-langchain \
    --description "Bedrock LangChain layer" \
    --license-info "MIT" \
    --content S3Bucket=${S3_ARTIFACT_BUCKET_NAME},S3Key=agent/lambda-layers/bedrock-langchain.zip \
    --compatible-runtimes python3.9 \
    --query LayerVersionArn --output text)

aws lambda update-function-configuration \
  --function-name arn:aws:lambda:us-east-1:174671970284:function:avivabedrockapp-GenAILexHandler \
  --layers ${BEDROCK_LANGCHAIN_LAYER_ARN} arn:aws:lambda:us-east-1:174671970284:layer:pypdf:1

