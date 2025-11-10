#!/bin/bash
# Helper script to check available modules on OSC Ascend

echo "======================================"
echo "OSC Ascend - Available Modules Check"
echo "======================================"
echo ""

echo "Checking Python modules..."
echo "----------------------------"
module avail python 2>&1 | grep -i python || module spider python 2>&1 | head -20
echo ""

echo "Checking CUDA modules..."
echo "----------------------------"
module avail cuda 2>&1 | grep -i cuda || module spider cuda 2>&1 | head -20
echo ""

echo "Checking currently loaded modules..."
echo "----------------------------"
module list
echo ""

echo "======================================"
echo "Usage:"
echo "  To load a module: module load <name>/<version>"
echo "  Example: module load python/3.10"
echo "  Example: module load cuda/12.1"
echo "======================================"
