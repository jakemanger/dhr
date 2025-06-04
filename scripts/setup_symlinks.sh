#!/bin/bash

echo "=== Symlink Setup ==="

prompt_for_path() {
    local label="$1"
    local path=""
    while true; do
        read -rp "Path to $label folder: " path
        path="${path%/}"
        # Remove quotes if present
        path="${path#\'}"
        path="${path%\'}"
        path="${path#\"}"
        path="${path%\"}"
        [ -z "$path" ] && echo "Path cannot be empty." && continue
        [ -d "$path" ] && break
        echo "Directory '$path' not found."
    done
    echo "$path"
}

create_symlink() {
    local src="$1"
    local link="$2"
    if [ -L "$link" ]; then
        rm "$link"
    elif [ -e "$link" ]; then
        read -rp "'$link' exists. Remove and link? (y/N): " resp
        [[ "$resp" =~ ^[Yy]$ ]] && rm -rf "$link" || return 1
    fi
    ln -s "$src" "$link" && echo "$link -> $src"
}

echo "1. Dataset"
dataset_path=$(prompt_for_path "dataset")
echo

echo "2. Logs"
logs_path=$(prompt_for_path "logs")
echo

echo "3. Output"
output_path=$(prompt_for_path "output")
echo

echo "Dataset: $dataset_path"
echo "Logs:    $logs_path"
echo "Output:  $output_path"
read -rp "Create these symlinks here? (y/N): " confirm

if [[ "$confirm" =~ ^[Yy]$ ]]; then
    create_symlink "$dataset_path" dataset
    create_symlink "$logs_path" logs
    create_symlink "$output_path" output
    echo "Symlinks created:"
    ls -l | grep "^l" | grep -E "(dataset|logs|output)"
else
    echo "Cancelled."
fi 