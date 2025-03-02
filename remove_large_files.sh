#!/bin/bash

# Script to remove large files from Git history
echo "Starting to remove large files from Git history..."

# Create a temporary branch to work on
git checkout --orphan temp_branch

# Add all files to the new branch except the large ones
echo "Adding files to the new branch (excluding large files)..."
git add --all -- ':!models/best_facial_model.pth' ':!models/final_facial_model.pth' ':!models/*.pth' ':!pa_stable_v190700_20210406.tgz'

# Commit the changes
git commit -m "Remove large files from history"

# Delete the main branch
echo "Removing old main branch..."
git branch -D main

# Rename the temporary branch to main
echo "Renaming temporary branch to main..."
git branch -m main

# Force push to remote repository
echo "Force pushing changes to remote repository..."
echo "WARNING: This will overwrite the remote repository's main branch!"
echo "Press Ctrl+C to cancel or Enter to continue..."
read

git push -f origin main

echo "Large files have been removed from Git history."
echo "Make sure your .gitignore file includes the large files to prevent them from being tracked again."