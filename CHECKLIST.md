# 📋 Pre-Deployment Checklist

Use this checklist before deploying your Small-World Networks website to GitHub Pages.

## ✅ File Preparation

- [ ] All graph images generated in `website/images/`
  - [ ] `regular_network.png`
  - [ ] `small_world_network.png`
  - [ ] `random_network.png`

- [ ] PowerPoint file copied to `website/` folder
  - [ ] `SmalWrld.pptx` is present

- [ ] All website files are present
  - [ ] `index.html`
  - [ ] `styles.css`
  - [ ] `script.js`
  - [ ] `README.md`
  - [ ] `deploy.sh`

## ✅ Code Verification

- [ ] Python code runs without errors
  ```bash
  python SmallWrld.py
  ```

- [ ] Graph generation works
  ```bash
  python generate_graphs.py
  ```

- [ ] All dependencies installed
  ```bash
  pip install -r requirements.txt
  ```

## ✅ Website Testing

- [ ] Website opens in browser locally
  ```bash
  cd website
  python3 -m http.server 8000
  # Visit http://localhost:8000
  ```

- [ ] Navigation works (smooth scroll)
- [ ] All images load correctly
- [ ] Code tabs switch properly
- [ ] Copy button works
- [ ] Download button for PowerPoint works
- [ ] Responsive on mobile (test with browser DevTools)
- [ ] No console errors (F12 → Console tab)

## ✅ GitHub Setup

- [ ] GitHub account created
- [ ] New repository created
- [ ] Repository is **Public** (required for free GitHub Pages)
- [ ] Git initialized in project folder
  ```bash
  git init
  ```

## ✅ Deployment

- [ ] All files committed
  ```bash
  git add .
  git commit -m "Initial commit: Small-World Networks"
  ```

- [ ] Connected to GitHub remote
  ```bash
  git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
  ```

- [ ] Pushed to GitHub
  ```bash
  git push -u origin main
  ```

- [ ] GitHub Pages enabled
  - Go to Settings → Pages
  - Source: `main` branch
  - Folder: `/website`
  - Save

- [ ] Website is live (wait 1-2 minutes)
  - Visit: `https://YOUR_USERNAME.github.io/YOUR_REPO/`

## ✅ Post-Deployment Updates

- [ ] Updated presentation viewer URL in `index.html`
  - Replace `YOUR_GITHUB_PAGES_URL` with actual URL
  
- [ ] Updated GitHub repository link in `index.html`
  - Replace `YOUR_USERNAME/YOUR_REPO` with actual values

- [ ] Tested live website
  - [ ] All sections load
  - [ ] Images display
  - [ ] Presentation viewer works (or download works)
  - [ ] Code copy works
  - [ ] Mobile responsive

- [ ] Committed post-deployment changes
  ```bash
  git add website/index.html
  git commit -m "Update URLs for production"
  git push
  ```

## ✅ Final Polish

- [ ] Updated README files with actual URLs
- [ ] Added repository description on GitHub
- [ ] Added GitHub Pages URL to repository description
- [ ] Verified all links work
- [ ] Spell-checked content
- [ ] Tested on different browsers (Chrome, Firefox, Safari)

## ✅ Share Your Work

- [ ] Shared link with professor
- [ ] Added to portfolio
- [ ] Shared on LinkedIn/social media (optional)
- [ ] Included in resume/CV (optional)

## 🎉 Launch!

Once all items are checked:

1. **Your website is live!**
2. **Share the URL**: `https://YOUR_USERNAME.github.io/YOUR_REPO/`
3. **Present with confidence!**

---

**Pro Tips**:

- Keep a screenshot of your website for backup
- Save the GitHub Pages URL somewhere safe
- Test on your phone before presenting
- Have the GitHub repository link ready to share
- Prepare to explain both the code and the website

**Common Issues**:

- **404 Error**: Make sure you selected `/website` folder, not root
- **Images not loading**: Check file names match exactly (case-sensitive)
- **Styling broken**: Clear browser cache (Cmd+Shift+R)
- **Presentation not showing**: Use the download button, or update the iframe URL

**Need Help?**: Check DEPLOYMENT_GUIDE.md for detailed troubleshooting
