#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

REMOTE="${GIT_REMOTE:-origin}"
BRANCH="${GIT_BRANCH:-$(git branch --show-current)}"
MESSAGE="${*:-Auto commit: $(date '+%Y-%m-%d %H:%M:%S')}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: $PROJECT_DIR is not a git repository."
  exit 1
fi

if [[ -z "$BRANCH" ]]; then
  echo "Error: cannot determine current git branch."
  exit 1
fi

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "Error: git remote '$REMOTE' does not exist."
  exit 1
fi

if git ls-files --error-unmatch config.yaml >/dev/null 2>&1; then
  echo "Error: config.yaml is tracked by git. Abort to avoid pushing local API secrets."
  exit 1
fi

echo "Repository: $PROJECT_DIR"
echo "Remote:     $REMOTE ($(git remote get-url "$REMOTE"))"
echo "Branch:     $BRANCH"

git add -A

if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git diff --cached --check
  echo "Committing: $MESSAGE"
  git commit -m "$MESSAGE"
fi

if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  git push
else
  git push -u "$REMOTE" "$BRANCH"
fi

echo "Done."
