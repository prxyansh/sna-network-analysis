# 🚀 Quick Start - Deploy in 5 Minutes

## For the Impatient (Fastest Path to Live Website)

### Step 1: Create GitHub Repository (2 minutes)

1. Go to https://github.com/new
2. Repository name: `small-world-networks` (or your choice)
3. Make it **Public**
4. Click "Create repository"

### Step 2: Push Your Code (1 minute)

```bash
cd /Users/prx./Documents/SNA

# Initialize and push
git init
git add .
git commit -m "Initial commit: Small-World Networks website"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

**Replace `YOUR_USERNAME` and `YOUR_REPO` with your actual GitHub username and repository name!**

### Step 3: Enable GitHub Pages (1 minute)

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Source: Select **main** branch
4. Folder: Select **/website** (important!)
5. Click **Save**

### Step 4: Update URLs (1 minute)

1. Wait 1-2 minutes for deployment
2. Your site will be at: `https://YOUR_USERNAME.github.io/YOUR_REPO/`
3. Open `website/index.html` in your editor
4. Find line ~77 and replace `YOUR_GITHUB_PAGES_URL` with your actual URL
5. Find the GitHub button and replace `YOUR_USERNAME/YOUR_REPO`
6. Save and push:

```bash
git add website/index.html
git commit -m "Update URLs"
git push
```

### Step 5: Share! 🎉

**Your website is live!**

Visit: `https://YOUR_USERNAME.github.io/YOUR_REPO/`

---

## Even Faster? Use the Script!

```bash
cd /Users/prx./Documents/SNA/website
./deploy.sh
```

Then follow the on-screen instructions.

---

## What You Get

✅ Beautiful, modern website  
✅ Interactive code display  
✅ Network visualizations  
✅ Downloadable PowerPoint  
✅ Mobile responsive  
✅ Professional design  

---

## Example URLs

If your username is `johndoe` and repo is `sna-project`:

- **Repository**: `https://github.com/johndoe/sna-project`
- **Website**: `https://johndoe.github.io/sna-project/`
- **Presentation**: `https://johndoe.github.io/sna-project/SmalWrld.pptx`

---

## Need More Help?

- **Detailed guide**: See `DEPLOYMENT_GUIDE.md`
- **Checklist**: See `CHECKLIST.md`
- **Project docs**: See `README.md`

---

## Test Locally First (Optional)

```bash
cd /Users/prx./Documents/SNA/website
python3 -m http.server 8000
```

Visit: `http://localhost:8000`

---

**That's it! You're done!** 🎉
