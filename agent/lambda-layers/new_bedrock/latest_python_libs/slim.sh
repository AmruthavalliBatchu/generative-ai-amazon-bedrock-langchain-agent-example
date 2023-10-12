#!/bin/bash

# Navigate to the root of your Python packages directory
cd python

# Remove __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove *.dist-info directories
find . -type d -name "*.dist-info" -exec rm -rf {} +

# Remove tests directories (optional; use with caution)
# Uncomment the line below if you want to remove them
# find . -type d -name "tests" -exec rm -rf {} +
