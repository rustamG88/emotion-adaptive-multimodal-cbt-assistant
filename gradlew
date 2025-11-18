#!/usr/bin/env sh
if command -v gradle >/dev/null 2>&1; then
  GRADLE_CMD="gradle"
else
  echo "Gradle is required to build this project." >&2
  exit 1
fi
exec "$GRADLE_CMD" "$@"
