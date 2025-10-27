# GitHub Setup Instructions

## Step 1: Create GitHub Repository
1. Go to: https://github.com/new
2. Repository name: `sna-network-analysis`
3. Description: "Social Network Analysis: Small-World Networks & Email Network Analysis"
4. Make it **Public** (required for GitHub Pages)
5. **DO NOT** check any initialization options
6. Click "Create repository"

## Step 2: Push Your Code

After creating the repository, run these commands:

```bash
cd /Users/prx./Documents/SNA

# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/sna-network-analysis.git

# Push the code
git push -u origin main
```

## Step 3: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** tab
3. Scroll down to **Pages** section (left sidebar)
4. Under **Source**, select:
   - Branch: `main`
   - Folder: `/website`
5. Click **Save**

## Step 4: Access Your Live Website

After 1-2 minutes, your website will be live at:
```
https://YOUR_USERNAME.github.io/sna-network-analysis/
```

## Troubleshooting

If you get authentication errors when pushing:
1. Use a Personal Access Token instead of password
2. Generate one at: https://github.com/settings/tokens
3. When prompted for password, paste the token

Or use SSH instead:
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/sna-network-analysis.git
```

## Your Repository is Ready!
All files committed and ready to push.
