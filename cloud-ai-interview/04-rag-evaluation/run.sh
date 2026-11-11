#!/bin/sh
set -eu
lesson_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$lesson_dir"
if [ "${1:-}" = "answer" ]; then
    export LLM_READ_TIMEOUT="${LLM_READ_TIMEOUT:-60}"
    export REQUEST_DEADLINE="${REQUEST_DEADLINE:-90}"
    exec uv run --project "$lesson_dir/../03-rag-datawhale" --locked --env-file ../02-http-service/.env python "$lesson_dir/lab.py" "$@"
else
    exec uv run --project "$lesson_dir/../03-rag-datawhale" --locked python "$lesson_dir/lab.py" "$@"
fi
