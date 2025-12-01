#!/bin/bash
# Commands to upload to GitHub
# Replace YOUR_USERNAME and YOUR_REPO_NAME with your actual values

echo "🚀 Setting up Git repository..."
cd "/Users/tiffany/Desktop/untitled folder/flight "

# Initialize git
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: Aviation Data Analytics Project"

# Rename branch to main
git branch -M main

echo ""
echo "✅ Repository initialized!"
echo ""
echo "📝 Next steps:"
echo "1. Create a new repository on GitHub.com"
echo "2. Copy the repository URL (e.g., https://github.com/username/repo-name.git)"
echo "3. Run these commands (replace with your actual URL):"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
echo "   git push -u origin main"
echo ""

