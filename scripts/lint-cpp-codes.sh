#!/bin/bash
# Pre-commit hook for formatting and tidy

# Format all staged C++ files
echo "Running clang-format... $IGNORE_CLANG_TIDY_ERROR"

if [ -n "${IGNORE_CLANG_TIDY_ERROR}" ]; then
    echo "----------------------------------"
    echo "clang-tidy errors will be ignored."
    echo "----------------------------------"
fi

STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACMR | grep -E '\.(cc|cpp|h)$')

for file in $STAGED_FILES; do
    clang-format -i  "$file" # uses .clang-format, but you can use `--style=Google` instead
    git add "$file"  # Re-add the modified files to the commit
done

echo "Running clang-tidy..."
FAILED=0

for file in $STAGED_FILES; do
    if ! clang-tidy "$file" -quiet --fix -p build/compile_commands.json; then
        echo "clang-tidy failed for $file"
        FAILED=1
    fi
done

# Exit with failure if clang-tidy detected issues
if [[ ($FAILED == 1) && (-z "${IGNORE_CLANG_TIDY_ERROR}") ]]; then
    echo "Pre-commit hook failed: Fix the clang-tidy errors before committing."
    exit 1
fi
