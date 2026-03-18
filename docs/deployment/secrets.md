# Deployment Secrets Configuration

This document lists all GitHub Secrets required for CI/CD deployment.

## Required Secrets

### Staging Environment

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `STAGING_HOST` | Staging server hostname or IP | `staging.example.com` |
| `STAGING_SSH_KEY` | SSH private key for staging server (PEM format) | `-----BEGIN RSA PRIVATE KEY-----...` |
| `STAGING_REDIS_URL` | Redis connection URL for staging | `redis://staging-redis:6379/0` |
| `STAGING_TRAINING_API_KEY` | Internal API key for staging Training API | `staging-training-key-xxx` |
| `STAGING_JWT_SECRET` | JWT signing key for staging | `staging-jwt-secret-xxx` |
| `STAGING_MLFLOW_URI` | MLflow tracking server URI for staging | `http://staging-mlflow:5000` |

### Production Environment

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `PROD_HOST` | Production server hostname or IP | `prod.example.com` |
| `PROD_SSH_KEY` | SSH private key for production server (PEM format) | `-----BEGIN RSA PRIVATE KEY-----...` |
| `PROD_REDIS_URL` | Redis connection URL for production | `redis://prod-redis:6379/0` |
| `PROD_TRAINING_API_KEY` | Internal API key for production Training API | `prod-training-key-xxx` |
| `PROD_JWT_SECRET` | JWT signing key for production | `prod-jwt-secret-xxx` |
| `PROD_MLFLOW_URI` | MLflow tracking server URI for production | `http://prod-mlflow:5000` |

### Shared Secrets

| Secret Name | Description | How to Obtain |
|-------------|-------------|---------------|
| `GITHUB_TOKEN` | GitHub token for container registry | Automatically provided by GitHub Actions |
| `DOCKER_TOKEN` | Docker registry token (optional, GITHUB_TOKEN preferred) | GitHub Personal Access Token with `repo` scope |

## Setting Up Secrets

### Via GitHub UI

1. Navigate to your repository on GitHub
2. Go to **Settings** > **Secrets and variables** > **Actions**
3. Click **New repository secret**
4. Add each secret with its corresponding value

### Via GitHub CLI

```bash
# Staging secrets
gh secret set STAGING_HOST --body "staging.example.com"
gh secret set STAGING_SSH_KEY --body "$(cat ~/.ssh/staging_key)"
gh secret set STAGING_REDIS_URL --body "redis://staging-redis:6379/0"
gh secret set STAGING_TRAINING_API_KEY --body "your-staging-key"
gh secret set STAGING_JWT_SECRET --body "your-staging-jwt-secret"
gh secret set STAGING_MLFLOW_URI --body "http://staging-mlflow:5000"

# Production secrets
gh secret set PROD_HOST --body "prod.example.com"
gh secret set PROD_SSH_KEY --body "$(cat ~/.ssh/prod_key)"
gh secret set PROD_REDIS_URL --body "redis://prod-redis:6379/0"
gh secret set PROD_TRAINING_API_KEY --body "your-prod-key"
gh secret set PROD_JWT_SECRET --body "your-prod-jwt-secret"
gh secret set PROD_MLFLOW_URI --body "http://prod-mlflow:5000"
```

## Server Setup Requirements

### Staging/Production Server Requirements

1. **Operating System**: Ubuntu 22.04 LTS or later
2. **Software**:
   - Docker (latest stable)
   - Docker Compose (optional)
   - OpenSSH server

3. **Network**:
   - SSH access (port 22)
   - Ports 8000-8003 open for API services

### SSH Key Generation

```bash
# Generate SSH key pair
ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/deploy_key

# Add public key to server
ssh-copy-id -i ~/.ssh/deploy_key.pub ubuntu@staging.example.com
```

### Security Best Practices

1. **Use separate keys per environment**: Don't reuse the same SSH key for staging and production
2. **Restrict SSH key permissions**: Use `ed25519` keys, not RSA for better security
3. **Rotate secrets regularly**: Update production secrets at least quarterly
4. **Use least privilege**: Create dedicated deploy users on target servers
5. **Enable SSH key passphrase**: Add an extra layer of security to SSH keys

## Environment Configuration

### Staging Deployment Flow

```
GitHub Actions (build) → GHCR (image registry) → Staging Server (docker run)
```

### Production Deployment Flow

```
GitHub Actions (build) → GHCR (image registry) → Production Server (blue-green deploy)
                                      ↓
                            Health Check & Verification
                                      ↓
                            Traffic Switch (if healthy)
                                      ↓
                            Old Version Cleanup (if healthy)
                            OR Rollback (if unhealthy)
```

## Troubleshooting

### Common Issues

1. **SSH Connection Failed**
   - Verify the SSH key is correctly added to the server
   - Check that the server IP/hostname is correct
   - Ensure the deploy user has proper permissions

2. **Docker Login Failed**
   - Verify GITHUB_TOKEN has proper permissions
   - Check that the repository is set to public OR you have GHCR access

3. **Health Check Failed**
   - Check container logs: `docker logs business-api-staging`
   - Verify environment variables are correctly set
   - Ensure ports are not blocked by firewall

4. **Rollback Triggered**
   - Check the deployment logs in GitHub Actions
   - Verify the previous image exists in GHCR
   - Manual intervention may be required if rollback fails
