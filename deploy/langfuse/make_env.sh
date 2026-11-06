#!/usr/bin/env bash
# Creates deploy/langfuse/langfuse.env with random secrets for a self-hosted Langfuse (never committed).
# Usage: bash deploy/langfuse/make_env.sh <admin-email> [public-url]   e.g. http://127.0.0.1:3000
# Prints the project keys to put into the app's .env, and nothing else secret.
set -euo pipefail
cd "$(dirname "$0")"
[ -e langfuse.env ] && { echo "langfuse.env exists, not overwritten"; exit 1; }
email="${1:?admin email}"; url="${2:-http://127.0.0.1:3000}"
r() { openssl rand -hex "$1"; }
pk="pk-lf-$(r 16)"; sk="sk-lf-$(r 24)"
umask 077
cat > langfuse.env <<ENV
NEXTAUTH_URL=${url}
NEXTAUTH_SECRET=$(r 32)
SALT=$(r 32)
ENCRYPTION_KEY=$(r 32)
POSTGRES_PASSWORD=$(r 24)
CLICKHOUSE_PASSWORD=$(r 24)
MINIO_ROOT_PASSWORD=$(r 24)
REDIS_AUTH=$(r 24)
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=${pk}
LANGFUSE_INIT_PROJECT_SECRET_KEY=${sk}
LANGFUSE_INIT_USER_EMAIL=${email}
LANGFUSE_INIT_USER_PASSWORD=$(r 12)
ENV
chmod 600 langfuse.env
echo "wrote $(pwd)/langfuse.env (mode 600). Admin password: see LANGFUSE_INIT_USER_PASSWORD in that file."
echo "Add to the app's .env:  MEDRAG_TRACING=langfuse  LANGFUSE_HOST=${url}"
echo "                         LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY = the two LANGFUSE_INIT_PROJECT_* values"
