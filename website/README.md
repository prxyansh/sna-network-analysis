# Small-World Networks - Interactive Website

A beautiful, modern website showcasing the Watts-Strogatz Small-World Network Model for Social Network Analysis.

## 🌐 Live Demo

Visit the live website: `https://YOUR_USERNAME.github.io/YOUR_REPO/`

## ✨ Features

- **📊 Interactive Presentation**: Embedded PowerPoint viewer with download option
- **💻 Source Code Display**: Clean, syntax-highlighted code with copy functionality
- **📈 Network Visualizations**: Beautiful graphs showing network evolution
- **🎨 Modern Design**: Minimal, responsive design with smooth animations
- **🚀 Fast & Lightweight**: Optimized for performance and mobile devices

## 🏗️ Project Structure

```
website/
├── index.html          # Main HTML file
├── styles.css          # Modern CSS styling
├── script.js           # Interactive JavaScript
├── SmalWrld.pptx       # PowerPoint presentation
├── images/             # Network visualizations
│   ├── regular_network.png
│   ├── small_world_network.png
│   └── random_network.png
└── README.md           # This file
```

## 🚀 Deployment to GitHub Pages

### Quick Deploy

1. **Create a GitHub repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Small-World Networks website"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Enable GitHub Pages**:
   - Go to your repository on GitHub
   - Navigate to **Settings** → **Pages**
   - Under **Source**, select **main** branch and **/website** folder
   - Click **Save**
   - Your site will be published at `https://YOUR_USERNAME.github.io/YOUR_REPO/`

3. **Update the presentation viewer**:
   - After deployment, edit `index.html`
   - Replace `YOUR_GITHUB_PAGES_URL` with your actual GitHub Pages URL
   - Example: `https://johndoe.github.io/small-world-networks`
   - Commit and push changes

### Using the Deployment Script

Make the deploy script executable and run it:

```bash
chmod +x deploy.sh
./deploy.sh
```

This will automatically:
- Add all changes
- Commit with timestamp
- Push to GitHub
- Display your GitHub Pages URL

## 📦 Technologies Used

- **HTML5**: Semantic, accessible markup
- **CSS3**: Modern gradients, animations, flexbox, grid
- **JavaScript**: Vanilla JS for interactivity
- **Python**: NetworkX for graph generation
- **Matplotlib**: Network visualizations
- **Google Fonts**: Inter & JetBrains Mono

## 🎨 Design Features

- **Gradient backgrounds** with smooth transitions
- **Smooth scroll** navigation
- **Responsive layout** for mobile, tablet, and desktop
- **Code syntax** highlighting
- **Tab switching** for code sections
- **Copy to clipboard** functionality
- **Lazy loading** images
- **Intersection Observer** animations
- **Glass morphism** effects

## 🧠 The Small-World Model

This website demonstrates the Watts-Strogatz Small-World Network Model, which explains:

- **High clustering** like regular networks
- **Short path lengths** like random networks
- The **"six degrees of separation"** phenomenon
- Real-world applications in social networks, neural networks, and more

## 📝 Local Development

To run locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO/website
   ```

2. **Open in browser**:
   - Simply open `index.html` in your browser
   - Or use a local server:
     ```bash
     python3 -m http.server 8000
     # Visit http://localhost:8000
     ```

## 🔧 Customization

### Change Network Parameters

Edit the graph generation script:

```python
N = 20      # Number of nodes
K = 4       # Nearest neighbors
P_small = 0.2  # Rewiring probability
```

### Update Colors

Modify CSS variables in `styles.css`:

```css
:root {
    --primary: #4A90E2;
    --accent: #E74C3C;
    --dark: #1A1A2E;
    /* ... more variables */
}
```

### Add More Sections

Add new sections in `index.html` following the existing pattern:

```html
<section id="new-section" class="section">
    <div class="container">
        <h2 class="section-title">New Section</h2>
        <!-- Your content -->
    </div>
</section>
```

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- **Duncan Watts & Steven Strogatz**: For the Small-World Network Model
- **NetworkX**: Python library for network analysis
- **Matplotlib**: Visualization library

## 📞 Contact

For questions or feedback about this project, please open an issue on GitHub.

---

**Built with ❤️ for Social Network Analysis**
