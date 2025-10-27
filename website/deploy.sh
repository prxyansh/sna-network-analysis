#!/bin/bash

# ========================================
# GitHub Pages Deployment Script
# for Small-World Networks Website
# ========================================

echo "🚀 Deploying Small-World Networks Website to GitHub Pages..."
echo ""

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "⚠️  Not a git repository. Initializing..."
    git init
    git branch -M main
    echo "✅ Git repository initialized"
fi

# Add all changes
echo "📦 Adding changes..."
git add .

# Commit with timestamp
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
git commit -m "Deploy: $TIMESTAMP"

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Go to your GitHub repository settings"
echo "   2. Navigate to Settings → Pages"
echo "   3. Set source to 'main' branch and '/website' folder"
echo "   4. Your site will be live at:"
echo "      https://YOUR_USERNAME.github.io/YOUR_REPO/"
echo ""
echo "   5. Don't forget to update the presentation viewer URL in index.html!"
echo ""
