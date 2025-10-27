# Small-World Networks - Social Network Analysis Project

A comprehensive project demonstrating the **Watts-Strogatz Small-World Network Model** with beautiful visualizations, interactive code, and a modern web presentation.

## 🌐 Live Website

**Visit the live demo**: `https://YOUR_USERNAME.github.io/YOUR_REPO/`

![Small-World Networks](website/images/small_world_network.png)

## 📋 Project Overview

This project explores the fascinating world of small-world networks, explaining why you're only "six degrees of separation" from anyone else on Earth. It includes:

- **Interactive Website**: Modern, responsive web interface
- **Python Implementation**: Complete Watts-Strogatz model
- **Network Visualizations**: Beautiful graphs showing network evolution
- **PowerPoint Presentation**: Comprehensive academic presentation
- **Educational Content**: Theory, implementation, and applications

## 📁 Project Structure

```
SNA/
├── website/                    # Web presentation
│   ├── index.html             # Main website
│   ├── styles.css             # Modern styling
│   ├── script.js              # Interactive features
│   ├── SmalWrld.pptx          # PowerPoint presentation
│   ├── images/                # Network visualizations
│   │   ├── regular_network.png
│   │   ├── small_world_network.png
│   │   └── random_network.png
│   ├── deploy.sh              # Deployment script
│   └── README.md              # Website documentation
│
├── SmallWrld.py               # Main implementation
├── generate_graphs.py         # Image generation script
├── requirements.txt           # Python dependencies
├── DEPLOYMENT_GUIDE.md        # Step-by-step deployment
└── README.md                  # This file
```

## 🚀 Quick Start

### Run the Python Code

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the interactive visualization**:
   ```bash
   python SmallWrld.py
   ```
   This will display three graphs showing the network evolution.

3. **Generate images for the website**:
   ```bash
   python generate_graphs.py
   ```

### Deploy the Website

1. **Test locally**:
   ```bash
   cd website
   python3 -m http.server 8000
   # Visit http://localhost:8000
   ```

2. **Deploy to GitHub Pages**:
   ```bash
   cd website
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Follow the detailed guide**: See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 🎯 Features

### Python Implementation
- ✅ Complete Watts-Strogatz algorithm
- ✅ Configurable parameters (N, K, p)
- ✅ Visual distinction between original and rewired edges
- ✅ Support for three network types (regular, small-world, random)

### Website Features
- ✅ Beautiful, minimal design with smooth animations
- ✅ Embedded PowerPoint presentation viewer
- ✅ Interactive code display with syntax highlighting
- ✅ Copy-to-clipboard functionality
- ✅ Responsive design (mobile, tablet, desktop)
- ✅ Network visualization gallery
- ✅ Educational content and explanations
- ✅ Download options for code and presentation

### Visualizations
- ✅ Regular Network (p=0): High clustering, long paths
- ✅ Small-World Network (p=0.2): The "sweet spot"
- ✅ Random Network (p=1): Low clustering, short paths

## 🧠 The Small-World Model

The Watts-Strogatz model demonstrates how real-world networks achieve both:

1. **High Clustering**: Like regular lattices (friends of friends are friends)
2. **Short Path Lengths**: Like random graphs (six degrees of separation)

This explains phenomena in:
- Social networks
- Neural networks
- Power grids
- The Internet
- Disease spread
- Information diffusion

### Algorithm Overview

```python
1. Start with a regular ring lattice (N nodes, each connected to K neighbors)
2. Rewire each edge with probability p:
   - Keep one endpoint fixed
   - Reconnect to a random node (avoiding self-loops and duplicates)
3. Result: Network interpolates between regular (p=0) and random (p=1)
```

### Key Parameters

- **N = 20**: Number of nodes in the network
- **K = 4**: Each node connects to K nearest neighbors initially
- **p = 0.2**: Rewiring probability (the "sweet spot")

## 🛠️ Technologies Used

### Backend
- **Python 3.13**
- **NetworkX**: Network analysis and graph generation
- **Matplotlib**: Network visualizations

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern gradients, animations, flexbox, grid
- **JavaScript (Vanilla)**: Interactivity and smooth UX
- **Google Fonts**: Inter & JetBrains Mono typography

## 📊 Included Files

### Code Files
- `SmallWrld.py`: Interactive visualization with three graphs
- `generate_graphs.py`: Batch image generation for website

### Website Files
- `index.html`: Complete website structure
- `styles.css`: Professional, modern styling (~700 lines)
- `script.js`: Interactive features and smooth UX
- `SmalWrld.pptx`: Academic presentation

### Documentation
- `README.md`: This project overview
- `DEPLOYMENT_GUIDE.md`: Detailed deployment instructions
- `website/README.md`: Website-specific documentation

## 🎨 Website Highlights

### Design Principles
- **Minimal**: Clean, distraction-free interface
- **Modern**: Gradients, animations, glass morphism
- **Accessible**: Semantic HTML, readable typography
- **Fast**: Optimized images, lazy loading
- **Responsive**: Works on all devices

### Interactive Features
- Smooth scroll navigation
- Tab switching for code sections
- Copy-to-clipboard for code
- Fade-in animations on scroll
- Navbar scroll effects
- Lazy loading images

## 📚 Academic Context

This project demonstrates understanding of:

- Graph theory fundamentals
- Network topology analysis
- Small-world phenomena
- Clustering coefficient
- Average path length
- Python scientific computing
- Data visualization
- Web development

## 🎓 Usage for Presentation

### For Your Professor
1. Share the live website URL
2. Highlight the interactive code display
3. Walk through the visualizations section
4. Download button for the PowerPoint
5. Explain the implementation details

### Presentation Flow
1. Start with the hero section (introduction)
2. Show the three visualizations
3. Explain the code implementation
4. Open the PowerPoint for detailed slides
5. Discuss real-world applications

## 🔧 Customization

### Change Network Parameters

Edit parameters in the Python files:

```python
N = 20       # Number of nodes
K = 4        # Nearest neighbors
P_small = 0.2   # Rewiring probability
```

### Update Website Colors

Modify CSS variables in `website/styles.css`:

```css
:root {
    --primary: #4A90E2;
    --accent: #E74C3C;
    --dark: #1A1A2E;
    /* ... more variables */
}
```

### Add More Sections

Follow the existing pattern in `index.html` and `styles.css`.

## 📦 Installation

### Prerequisites
- Python 3.7+ (tested with 3.13)
- pip package manager
- Git (for deployment)
- Modern web browser

### Setup

```bash
# Clone or download the project
cd /Users/prx./Documents/SNA

# Create virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the code
python SmallWrld.py

# Generate website images
python generate_graphs.py

# Test website locally
cd website
python3 -m http.server 8000
# Visit http://localhost:8000
```

## 🚀 Deployment

See the detailed [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for complete instructions.

Quick deploy:
```bash
cd website
chmod +x deploy.sh
./deploy.sh
```

## 📝 References

- **Original Paper**: Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks. *Nature*, 393(6684), 440-442.
- **NetworkX Documentation**: https://networkx.org/
- **Small-World Network (Wikipedia)**: https://en.wikipedia.org/wiki/Small-world_network

## 🙏 Acknowledgments

- **Duncan Watts & Steven Strogatz**: For the groundbreaking small-world model
- **NetworkX Team**: For the excellent graph library
- **Python Community**: For amazing scientific computing tools

## 📄 License

This project is open source and available for educational purposes.

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the DEPLOYMENT_GUIDE.md
- Review the website README.md

---

**Built with ❤️ for Social Network Analysis**

*Project created October 2025*
