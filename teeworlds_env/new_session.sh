#!/usr/bin/env bash

CONFIG_FILE="./config.yaml"

if [ "$#" -ne 1 ]; then
  echo "ERROR: need session name as first cli argument"
  exit 1
fi

SESSION_NAME="$1"
GIT_BRANCH="model/${SESSION_NAME}"

echo "creating session: \"$SESSION_NAME\""
echo "creating git branch: \"$GIT_BRANCH\""

SESSION_EXISTS=$(git ls-remote --heads https://github.com/dephiloper/independent-coursework-rl "$GIT_BRANCH")

if [ -n "$SESSION_EXISTS" ]; then
  echo "ERROR: branch \"$GIT_BRANCH\" already existing"
  exit 1
fi

# git checkout -b "$GIT_BRANCH"

echo "rewriting config file"

# remove session_name from config
sed -i '/session_name/d' "$CONFIG_FILE"

echo "session_name: \"${SESSION_NAME}\"" >> "$CONFIG_FILE"
