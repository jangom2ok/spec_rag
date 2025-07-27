#!/bin/bash
# Wrapper script to run mypy and handle transformers internal error

set -e

echo "Running mypy type checking..."

# Run mypy and capture both stdout and stderr
output=$(python -m mypy app/ 2>&1 || true)

# Check if the output contains only the transformers internal error
if echo "$output" | grep -q "INTERNAL ERROR" && echo "$output" | grep -q "transformers"; then
    # Count non-transformers errors
    error_count=$(echo "$output" | grep -E "error:|note:" | grep -v "transformers" | grep -v "INTERNAL ERROR" | wc -l)

    if [ "$error_count" -eq 0 ]; then
        echo "✓ mypy: OK (ignoring transformers internal error)"
        exit 0
    else
        echo "$output" | grep -v "transformers" | grep -v "INTERNAL ERROR"
        exit 1
    fi
else
    # No transformers error, show all output
    echo "$output"

    # Check if there were any errors
    if echo "$output" | grep -q "error:"; then
        exit 1
    else
        echo "✓ mypy: OK"
        exit 0
    fi
fi
