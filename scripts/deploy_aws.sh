#!/usr/bin/env bash
set -euo pipefail

AWS_REGION=${AWS_REGION:-us-east-1}
ECR_REPO_API=${ECR_REPO_API:-incident-intel-api}
ECR_REPO_FRONTEND=${ECR_REPO_FRONTEND:-incident-intel-frontend}

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker build -f docker/dockerfiles/backend.Dockerfile -t "$ECR_REPO_API:latest" .
docker build -f docker/dockerfiles/frontend.Dockerfile -t "$ECR_REPO_FRONTEND:latest" frontend

docker tag "$ECR_REPO_API:latest" "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_API:latest"
docker tag "$ECR_REPO_FRONTEND:latest" "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_FRONTEND:latest"

docker push "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_API:latest"
docker push "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_FRONTEND:latest"

kubectl apply -f infra/k8s/data-services.yaml
kubectl apply -f infra/k8s/api-deployment.yaml
kubectl apply -f infra/k8s/frontend-deployment.yaml
