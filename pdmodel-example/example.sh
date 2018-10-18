#!/bin/bash
#
# Docker build helper.

# immediately exit if a command fails
set -e

PROJECT_NAME="pdmodel-example"

USAGE="$0 <command>"
COMMAND="$1"
if [ "$COMMAND" == "" ]; then
    echo "$USAGE"
	  exit 1
fi

build () {
    docker build -t "$PROJECT_NAME" .
}

run () {
    docker run -it --rm "$PROJECT_NAME" "$@"
}

case $COMMAND in
    "build")
        build
        ;;
    "run")
        run "$@"
        ;;
    *)
        echo "Command not found."
        exit 1
        ;;
esac
