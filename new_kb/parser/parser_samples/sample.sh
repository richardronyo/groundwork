#!/usr/bin/env bash
# Sample script exercising functions, conditionals, loops, sourcing.
source ./helpers.sh

greet() {
  local name="$1"
  if [[ -z "$name" ]]; then
    echo "no name given"
    return 1
  fi
  echo "hello, $name"
}

for user in alice bob; do
  greet "$user"
done
