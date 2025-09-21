#!/bin/bash

# Check if filename is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename.tex>"
    echo "Example: $0 sdu.talk.tex"
    exit 1
fi

# Set the working directory
WORK_DIR="$(dirname "$0")"
cd "$WORK_DIR" || exit 1

# Set the presentation file from argument
PRESENTATION="$1"

# Function to clean up auxiliary files
cleanup() {
    echo "Cleaning up auxiliary files..."
    # Clean current directory
    rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb
}

# Function to handle errors
handle_error() {
    echo "Error: $1"
    cleanup
    exit 1
}

# Check if xelatex is installed
if ! command -v xelatex &> /dev/null; then
    handle_error "xelatex is not installed. Please install a TeX distribution with XeTeX support."
fi

# Check if the presentation file exists
if [ ! -f "$PRESENTATION" ]; then
    handle_error "Presentation file not found: $PRESENTATION"
fi

# Compile the presentation
echo "Compiling presentation $1..."
if ! xelatex -interaction=nonstopmode -halt-on-error "$PRESENTATION"; then
    handle_error "First compilation failed"
fi

if ! xelatex -interaction=nonstopmode -halt-on-error "$PRESENTATION"; then
    handle_error "Second compilation failed"
fi

# Clean up auxiliary files
cleanup

echo "Compilation successful! The PDF has been generated."
