#!/usr/bin/env bash

names=(
"Mayank Shandilya"
"Tejal Khatri"
"SUNU JOHNSON"
"Devendra Gupta"
"Roshan Shrestha"
)

for name in "${names[@]}"; do
  folder_name="${name// /_}"
  mkdir -p "$folder_name"
  touch "$folder_name/README.md"
  echo "Created folder: $folder_name"
done
