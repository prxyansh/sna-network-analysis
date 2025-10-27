// ========================================
// SMALL-WORLD NETWORKS - INTERACTIVE JS
// ========================================

// Project Tab Switching
const projectTabs = document.querySelectorAll('.project-tab');
const projectContents = document.querySelectorAll('.project-content');

projectTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs
        projectTabs.forEach(t => t.classList.remove('active'));
        // Add active to clicked tab
        tab.classList.add('active');
        
        // Hide all project contents
        projectContents.forEach(content => content.classList.remove('active'));
        
        // Show selected project content
        const projectId = tab.getAttribute('data-project') + '-project';
        const targetProject = document.getElementById(projectId);
        if (targetProject) {
            targetProject.classList.add('active');
            // Scroll to top of project content
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
    });
});

// Create animated particles in hero section
function createParticles() {
    const hero = document.querySelector('.hero');
    if (!hero) return;
    
    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.cssText = `
            position: absolute;
            width: ${Math.random() * 4 + 2}px;
            height: ${Math.random() * 4 + 2}px;
            background: radial-gradient(circle, rgba(255,255,255,0.8), transparent);
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: float ${Math.random() * 10 + 10}s linear infinite;
            opacity: ${Math.random() * 0.5 + 0.3};
            pointer-events: none;
        `;
        hero.appendChild(particle);
    }
}

// Add floating animation
const style = document.createElement('style');
style.textContent = `
    @keyframes float {
        0%, 100% {
            transform: translateY(0) translateX(0);
        }
        25% {
            transform: translateY(-20px) translateX(10px);
        }
        50% {
            transform: translateY(-40px) translateX(-10px);
        }
        75% {
            transform: translateY(-20px) translateX(5px);
        }
    }
`;
document.head.appendChild(style);

// Initialize particles when DOM loads
document.addEventListener('DOMContentLoaded', createParticles);

// Smooth scroll for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const offset = 80; // Account for fixed navbar
            const targetPosition = target.offsetTop - offset;
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });
        }
    });
});

// Code tab switching
const codeTabs = document.querySelectorAll('.code-tab');
const codeContents = document.querySelectorAll('.code-content');

codeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active class from all tabs
        codeTabs.forEach(t => t.classList.remove('active'));
        // Add active class to clicked tab
        tab.classList.add('active');
        
        // Hide all code contents
        codeContents.forEach(content => content.classList.add('hidden'));
        
        // Show the selected content
        const targetId = tab.getAttribute('data-tab');
        const targetContent = document.getElementById(targetId);
        if (targetContent) {
            targetContent.classList.remove('hidden');
        }
    });
});

// Copy code functionality
function copyCode(codeElementId) {
    // If a specific code element ID is provided, use it; otherwise find the active one
    let codeElement;
    if (codeElementId) {
        codeElement = document.getElementById(codeElementId);
    } else {
        const activeContent = document.querySelector('.code-content:not(.hidden)');
        if (activeContent) {
            codeElement = activeContent.querySelector('code');
        }
    }
    
    if (!codeElement) return;
    
    const codeText = codeElement.textContent;
    
    // Create temporary textarea to copy text
    const textarea = document.createElement('textarea');
    textarea.value = codeText;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    
    try {
        document.execCommand('copy');
        
        // Visual feedback - find the closest copy button
        const copyBtn = event ? event.target.closest('.btn-copy') : document.querySelector('.btn-copy');
        if (copyBtn) {
            const originalHTML = copyBtn.innerHTML;
            copyBtn.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                Copied!
            `;
            const originalBg = copyBtn.style.background;
            copyBtn.style.background = 'rgba(76, 175, 80, 0.3)';
            
            setTimeout(() => {
                copyBtn.innerHTML = originalHTML;
                copyBtn.style.background = originalBg;
            }, 2000);
        }
    } catch (err) {
        console.error('Failed to copy code:', err);
    }
    
    document.body.removeChild(textarea);
}

// Navbar scroll effect
let lastScroll = 0;
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > 100) {
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.05)';
    }
    
    lastScroll = currentScroll;
});

// Intersection Observer for fade-in animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
            // Add stagger effect
            const delay = entry.target.dataset.delay || 0;
            entry.target.style.animationDelay = `${delay}ms`;
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe elements for animation with stagger
const animatedElements = document.querySelectorAll('.viz-card, .presentation-card, .stat-card');
animatedElements.forEach((el, index) => {
    el.dataset.delay = index * 100;
    observer.observe(el);
});

// Add subtle hover effect on cards (removed 3D tilt for minimal look)

// Add loading state to images
document.querySelectorAll('img[loading="lazy"]').forEach(img => {
    img.addEventListener('load', function() {
        this.style.opacity = '1';
        this.style.transition = 'opacity 0.3s ease';
    });
    img.style.opacity = '0';
});

// Handle presentation iframe loading
window.addEventListener('DOMContentLoaded', () => {
    const presentationViewer = document.querySelector('.presentation-viewer iframe');
    if (presentationViewer) {
        const currentURL = window.location.href;
        const baseURL = currentURL.substring(0, currentURL.lastIndexOf('/'));
        
        // Update iframe src if deployed (not localhost)
        if (!window.location.hostname.includes('localhost') && !window.location.hostname.includes('127.0.0.1')) {
            const newSrc = presentationViewer.src.replace('YOUR_GITHUB_PAGES_URL', baseURL);
            presentationViewer.src = newSrc;
        }
    }
});

// Console easter egg
console.log('%c Small-World Networks 🌐', 'color: #4A90E2; font-size: 24px; font-weight: bold;');
console.log('%c Exploring the Watts-Strogatz Model', 'color: #667eea; font-size: 14px;');
console.log('%c Built with ❤️ for Social Network Analysis', 'color: #7C8B9E; font-size: 12px;');

// Download all reports functionality
function downloadAllReports() {
    const files = [
        'project_finl_outputs/report_final.pdf',
        'project_finl_outputs/link_analysis.csv',
        'project_finl_outputs/node_classification_labels.csv',
        'project_finl_outputs/influence_scores.csv',
        'project_finl_outputs/anomalies.csv',
        'project_finl_outputs/link_prediction_scores.csv',
        'project_finl_outputs/link_prediction_auc.csv',
        'project_finl_outputs/graph_attributes.gexf'
    ];
    
    const button = event.target.closest('.btn-download-all');
    const originalHTML = button.innerHTML;
    
    button.innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
        <span>Downloading...</span>
    `;
    button.disabled = true;
    
    // Download each file with a small delay to avoid browser blocking
    files.forEach((file, index) => {
        setTimeout(() => {
            const a = document.createElement('a');
            a.href = file;
            a.download = file.split('/').pop();
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            
            // Show completion after last file
            if (index === files.length - 1) {
                setTimeout(() => {
                    button.innerHTML = `
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="20 6 9 17 4 12"></polyline>
                        </svg>
                        <span>Downloaded!</span>
                    `;
                    
                    setTimeout(() => {
                        button.innerHTML = originalHTML;
                        button.disabled = false;
                    }, 3000);
                }, 500);
            }
        }, index * 300); // 300ms delay between downloads
    });
}

// Performance monitoring (optional)
if ('performance' in window) {
    window.addEventListener('load', () => {
        const perfData = performance.getEntriesByType('navigation')[0];
        if (perfData) {
            console.log(`Page loaded in ${(perfData.loadEventEnd - perfData.fetchStart).toFixed(2)}ms`);
        }
    });
}
