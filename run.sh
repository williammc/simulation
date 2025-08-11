#!/bin/bash

# SLAM Simulation System - Environment Setup and Runner
# This script manages the virtual environment and forwards commands to the Python CLI

set -e  # Exit on error

# Get hostname for environment naming
HOSTNAME=$(hostname -s | tr '[:upper:]' '[:lower:]' | tr -d ' ')
VENV_NAME=".venv-${HOSTNAME}"

# Python command
PYTHON=${PYTHON:-python3}

# Color output (only for setup messages)
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Setup or activate virtual environment
setup_venv() {
    if [ ! -d "$VENV_NAME" ]; then
        echo -e "${YELLOW}Creating virtual environment: ${VENV_NAME}${NC}"
        $PYTHON -m venv "$VENV_NAME"
        
        # Activate and install dependencies
        source "$VENV_NAME/bin/activate"
        echo -e "${YELLOW}Installing dependencies...${NC}"
        pip install --quiet --upgrade pip
        echo -e "${GREEN}Environment ready: ${VENV_NAME}${NC}"
    else
        # Just activate existing environment
        source "$VENV_NAME/bin/activate"
    fi
    pip install --quiet -r requirements.txt
}

# Main execution
main() {
    # Always setup/activate the virtual environment
    setup_venv
    
    # Forward all arguments to the Python CLI
    # The CLI handles all command logic, help, etc.
    $PYTHON -m tools.cli "$@"
}

# Check if being sourced or executed
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi