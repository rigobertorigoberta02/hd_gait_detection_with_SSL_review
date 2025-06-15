#!/bin/bash
NEWBASE="${1:-/content/drive/MyDrive/hd_data}"

# Find all Python files excluding .git directory
find . -path ./.git -prune -o -name "*.py" -type f -print | while read -r file; do
    sed -i.bak -e "s|/mlwell-data2|$NEWBASE|g" -e "s|/home/dafnas1|$NEWBASE|g" "$file"
done

grep -n -r '/mlwell-data2\|/home/dafnas1' . --exclude-dir=.git
if [ $? -eq 0 ]; then
    echo "ERROR: Some paths were not replaced." >&2
    exit 1
else
    echo "All paths successfully replaced."
fi
