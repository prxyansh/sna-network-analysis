# 🚀 Website Deployment Guide

## Website Overview

Your SNA presentation website showcases **TWO complete projects**:

### 1. Small-World Networks (Watts-Strogatz Model)
- PowerPoint presentation embedded
- Network visualizations (regular, small-world, random)
- Complete Python source code
- Analysis and findings

### 2. Email Network Analysis (Email-EU-Core Dataset)
- **1,005 nodes** and **25,571 edges**
- Comprehensive network overview with stats
- **Link Analysis**: PageRank, Eigenvector centrality
- **Community Detection**: Label Propagation algorithm
- **Influence Analysis**: PageRank + Betweenness centrality
- **Anomaly Detection**: IsolationForest on network features
- **Link Prediction**: Adamic-Adar, Jaccard, Preferential Attachment
- Complete Python implementation
- Downloadable reports (PDF, DOCX, GEXF)
- All visualizations and CSV data

## 📁 Project Structure

```
website/
├── index.html                    # Main website file
├── styles.css                    # All styling (1300+ lines)
├── script.js                     # Interactive features
├── DEPLOYMENT.md                 # This file
├── images/                       # Small-World visualizations
│   ├── regular_network.png
│   ├── small_world_network.png
│   └── random_network.png
└── project_finl_outputs/         # Email Network Analysis outputs (24 files)
    ├── subgraph.png
    ├── degree_distribution.png
    ├── link_analysis_top_pagerank.png
    ├── link_analysis_view.png
    ├── community_view.png
    ├── influence_top.png
    ├── influence_view.png
    ├── anomaly_scatter.png
    ├── anomaly_view.png
    ├── link_prediction_roc.png
    ├── link_prediction_roc_final.png
    ├── link_analysis.csv
    ├── node_classification_labels.csv
    ├── influence_scores.csv
    ├── anomalies.csv
    ├── link_prediction_scores.csv
    ├── link_prediction_auc.csv
    ├── graph_attributes.gexf
    ├── report.pdf
    ├── report.docx
    ├── report_final.pdf
    ├── MLCT2.pdf
    └── MLF.pdf
```

## 🌐 Deploy to GitHub Pages

### Step 1: Create GitHub Repository

```bash
# Navigate to your SNA directory
cd /Users/prx./Documents/SNA

# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Add SNA website with dual-project showcase"

# Create a repository on GitHub (https://github.com/new)
# Name it: sna-presentation or similar
# Then link and push:
git remote add origin https://github.com/YOUR_USERNAME/sna-presentation.git
git branch -M main
git push -u origin main
```

### Step 2: Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** tab
3. Scroll to **Pages** section (left sidebar)
4. Under **Source**, select:
   - Branch: `main`
   - Folder: `/website` (if website is in a subfolder) OR `/` (root)
5. Click **Save**
6. Wait 1-2 minutes for deployment

### Step 3: Access Your Live Website

Your website will be available at:
```
https://YOUR_USERNAME.github.io/sna-presentation/
```

Or if using a custom repository name:
```
https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/
```

## ✅ Pre-Deployment Checklist

- [x] All 3 Small-World network images generated
- [x] All 24 Email Network output files copied
- [x] HTML structure complete with dual-project tabs
- [x] CSS styling added (1300+ lines)
- [x] JavaScript interactions working
- [x] Project tab switching functional
- [x] Code copy buttons working
- [x] Download links for PDFs and CSVs
- [x] Author credentials in footer

## 🎨 Website Features

### Interactive Elements
- ✨ Floating particles animation in hero sections
- 🔄 Smooth tab switching between projects
- 📋 One-click code copying
- 🖱️ Minimal hover effects (per your request)
- 📱 Fully responsive design
- 🎨 Beautiful gradient color scheme

### Design Highlights
- **Colors**: Purple (#6366f1), Pink (#ec4899), Orange (#f97316)
- **Fonts**: Inter (main), JetBrains Mono (code)
- **Effects**: Glass morphism, gradients, smooth animations
- **Layout**: CSS Grid + Flexbox for responsive design

## 🧪 Local Testing

Your website is currently running at:
```
http://localhost:8080/website/
```

The server is running with PID: 24049

To restart the server:
```bash
cd /Users/prx./Documents/SNA/website
python3 -m http.server 8080
```

## 📝 Author Information

**Priyansh Kumar Paswan**  
Roll Number: 205124071  
Institution: NIT Tiruchirappalli  
Course: Social Network Analysis - 3rd Semester Assignment

## 🔧 Troubleshooting

### Images not loading after deployment
- Verify all image paths use relative URLs (no leading `/`)
- Check that `project_finl_outputs/` folder is committed to git
- Ensure all PNG files are in the correct directories

### PDFs not downloading
- Make sure PDF files are committed to GitHub
- Check file size limits (GitHub has 100MB per file limit)
- Large files may need Git LFS

### PowerPoint not showing
- Office365 iframe viewer requires public HTTPS URL
- Will work automatically once deployed to GitHub Pages
- Local preview won't show PowerPoint embed

## 📊 Content Summary

### Project 1: Small-World Networks
- Watts-Strogatz model implementation
- Parameters: N=20 nodes, K=4 neighbors, p=0.2 rewiring
- 3 network comparisons: Regular → Small-World → Random
- Complete source code with visualization

### Project 2: Email Network Analysis
- Dataset: Email-EU-Core network
- **Network Stats**: 1,005 nodes, 25,571 edges, avg degree 50.88
- **6 Major Analysis Tasks**:
  1. Network Overview & Statistics
  2. Link Analysis (PageRank, Eigenvector)
  3. Community Detection (Label Propagation)
  4. Influence Scoring (PageRank + Betweenness)
  5. Anomaly Detection (IsolationForest)
  6. Link Prediction (ML-based)
- **Outputs**: 12 visualizations, 6 CSV datasets, 5 PDF reports

## 🎯 Next Steps

1. Test the website locally in your browser
2. Push to GitHub repository
3. Enable GitHub Pages
4. Share your live URL!

---

**Built with ❤️ for Social Network Analysis**  
*Modern, Interactive, Professional*
