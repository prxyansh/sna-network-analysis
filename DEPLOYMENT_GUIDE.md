# 🚀 GitHub Pages Deployment Guide

## Step-by-Step Deployment Instructions

### 1️⃣ Prepare Your GitHub Repository

1. **Create a new repository on GitHub**:
   - Go to https://github.com/new
   - Name it something like `small-world-networks` or `sna-presentation`
   - Keep it **Public** (required for free GitHub Pages)
   - Don't initialize with README (we already have one)

### 2️⃣ Initialize Git (if not already done)

From your project root (`/Users/prx./Documents/SNA`):

```bash
cd /Users/prx./Documents/SNA
git init
git add .
git commit -m "Initial commit: Small-World Networks project"
git branch -M main
```

### 3️⃣ Connect to GitHub

Replace `YOUR_USERNAME` and `YOUR_REPO` with your actual GitHub username and repository name:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 4️⃣ Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** (top menu)
3. Scroll down and click **Pages** (left sidebar)
4. Under **Source**:
   - Branch: Select `main`
   - Folder: Select `/website` (this is important!)
   - Click **Save**
5. Wait 1-2 minutes for deployment

Your site will be live at:
```
https://YOUR_USERNAME.github.io/YOUR_REPO/
```

### 5️⃣ Update Presentation Viewer URL

After deployment, you need to update one line in `index.html`:

1. Open `website/index.html`
2. Find this line (around line 77):
   ```html
   src="https://view.officeapps.live.com/op/embed.aspx?src=YOUR_GITHUB_PAGES_URL/SmalWrld.pptx"
   ```
3. Replace `YOUR_GITHUB_PAGES_URL` with your actual URL:
   ```html
   src="https://view.officeapps.live.com/op/embed.aspx?src=https://YOUR_USERNAME.github.io/YOUR_REPO/SmalWrld.pptx"
   ```
4. Save and commit:
   ```bash
   git add website/index.html
   git commit -m "Update presentation viewer URL"
   git push
   ```

### 6️⃣ Update GitHub Repository Link

In `website/index.html`, find the "View on GitHub" button and update:

```html
<a href="https://github.com/YOUR_USERNAME/YOUR_REPO" target="_blank" class="btn btn-github">
```

### 7️⃣ Verify Deployment

1. Visit your GitHub Pages URL
2. Check that all sections load properly
3. Test the presentation download
4. Verify all images display correctly
5. Test code copy functionality
6. Check on mobile devices

## 🔧 Troubleshooting

### Issue: Site shows 404

**Solution**: 
- Make sure you selected `/website` as the folder, not `/` (root)
- Wait a few minutes after enabling Pages
- Check that `index.html` is in the `website/` folder

### Issue: Presentation doesn't load

**Solution**:
- Verify you updated the iframe src URL
- The PowerPoint file must be accessible at your GitHub Pages URL
- Try the download button - if it works, the file is accessible

### Issue: Images don't load

**Solution**:
- Check that images are in `website/images/` folder
- Verify file names match exactly (case-sensitive)
- Regenerate images with: `python3 generate_graphs.py`

### Issue: Styling looks broken

**Solution**:
- Clear browser cache (Cmd+Shift+R on Mac, Ctrl+Shift+R on Windows)
- Check browser console for errors (F12)
- Verify `styles.css` is in the `website/` folder

## 📱 Testing Locally Before Deployment

Run a local server to test:

```bash
cd website
python3 -m http.server 8000
```

Then visit: `http://localhost:8000`

## 🔄 Updating Your Site

After making changes:

```bash
# From the SNA directory
git add .
git commit -m "Update: description of changes"
git push

# Or use the deployment script
cd website
chmod +x deploy.sh
./deploy.sh
```

Changes will be live in 1-2 minutes.

## 🎨 Custom Domain (Optional)

If you have your own domain:

1. In GitHub Settings → Pages
2. Enter your custom domain
3. Update your DNS settings:
   - Add CNAME record pointing to `YOUR_USERNAME.github.io`
4. Wait for DNS propagation (up to 24 hours)

## 📊 Analytics (Optional)

Add Google Analytics:

1. Get your GA tracking ID
2. Add to `<head>` in `index.html`:
   ```html
   <!-- Google Analytics -->
   <script async src="https://www.googletagmanager.com/gtag/js?id=GA_TRACKING_ID"></script>
   <script>
     window.dataLayer = window.dataLayer || [];
     function gtag(){dataLayer.push(arguments);}
     gtag('js', new Date());
     gtag('config', 'GA_TRACKING_ID');
   </script>
   ```

## ✅ Post-Deployment Checklist

- [ ] Repository created and pushed to GitHub
- [ ] GitHub Pages enabled with `/website` folder
- [ ] Site is accessible at GitHub Pages URL
- [ ] Presentation viewer URL updated in HTML
- [ ] GitHub repository link updated in HTML
- [ ] All images loading correctly
- [ ] Code copy functionality working
- [ ] Smooth scrolling working
- [ ] Tab switching working
- [ ] Responsive on mobile devices
- [ ] Presentation downloads correctly
- [ ] README updated with actual URLs

## 🎉 Share Your Work!

Once deployed, share your website:

- Add the link to your GitHub repository description
- Share on LinkedIn, Twitter, etc.
- Include in your academic portfolio
- Submit to your professor with the live link

---

**Need help?** Check the GitHub Pages documentation: https://docs.github.com/pages
