# Portfolio Documentation Usage Guide

This repository now includes comprehensive portfolio documentation for the Overnight Map project.

## Files Added

### 1. `portfolio.json`
A machine-readable JSON file containing structured project information:
- Project metadata (name, tagline, description, URLs)
- Complete feature list
- Detailed tech stack breakdown
- Technical highlights and key metrics
- Challenges solved and solutions implemented
- Future enhancement roadmap
- Screenshot URLs

**Use cases:**
- Import into portfolio website databases
- Generate portfolio cards programmatically
- Feed into static site generators
- API responses for portfolio queries

### 2. `PORTFOLIO.md`
A comprehensive human-readable markdown document with:
- Detailed project overview and features
- Complete technical architecture description
- Data flow diagrams
- Code snippets and examples
- Challenges and solutions narrative
- Screenshots with descriptions
- Learning outcomes

**Use cases:**
- Display on portfolio websites
- Convert to HTML for web pages
- Include in technical presentations
- Share with potential employers or collaborators

## How to Use These Files

### For Portfolio Websites

**Static Site Generators (Hugo, Jekyll, Next.js, etc.):**
```javascript
// Example: Import portfolio.json in Next.js
import portfolioData from './portfolio.json';

export function OvernightMapProject() {
  const project = portfolioData.project;
  return (
    <div>
      <h1>{project.name}</h1>
      <p>{project.tagline}</p>
      <a href={project.live_url}>View Live</a>
      {/* Render features, tech stack, etc. */}
    </div>
  );
}
```

**Markdown Rendering:**
```python
# Example: Convert PORTFOLIO.md to HTML
import markdown

with open('PORTFOLIO.md', 'r') as f:
    md_content = f.read()
    
html = markdown.markdown(md_content)
```

### For Documentation

The PORTFOLIO.md file can be:
- Linked directly from README.md
- Converted to PDF for presentations
- Used as a template for other project documentation
- Shared with potential employers or recruiters

## Integration Examples

### Example 1: Portfolio Website Card
```html
<!-- Using data from portfolio.json -->
<div class="project-card">
  <h2>Overnight Map</h2>
  <p class="tagline">24/5 Stock Market Heat Map Visualization</p>
  <div class="tech-stack">
    <span class="tech">Python</span>
    <span class="tech">Dash</span>
    <span class="tech">Plotly</span>
    <span class="tech">GCP</span>
  </div>
  <a href="https://www.247map.app" class="btn">View Live</a>
  <a href="https://github.com/p-o-f/overnight-map" class="btn">View Code</a>
</div>
```

### Example 2: Resume/CV Entry
```
Overnight Map (2024)
• Developed real-time financial data visualization tool serving 600+ stocks
• Implemented async architecture handling 1,800+ API calls every 14 minutes
• Deployed on GCP with 24/5 uptime and mobile-responsive interface
• Technologies: Python, Dash, Plotly, aiohttp, Pandas, GCP

Live: https://www.247map.app
```

### Example 3: README Quick Link
Add to README.md:
```markdown
## Portfolio Documentation

For detailed information about this project including architecture, technical challenges, 
and implementation details, see [PORTFOLIO.md](PORTFOLIO.md).

For structured data suitable for portfolio websites, see [portfolio.json](portfolio.json).
```

## File Structure

```
overnight-map/
├── README.md              # Project README
├── PORTFOLIO.md           # Detailed portfolio documentation (this file can be used directly)
├── portfolio.json         # Structured project data
├── PORTFOLIO_USAGE.md     # This usage guide
├── main.py                # Production application
├── my_app.py              # Local development version
├── requirements.txt       # Python dependencies
└── assets/                # Static assets
    ├── favicon.ico
    └── style.css
```

## Quick Reference

| What You Need | Use This File |
|---------------|---------------|
| Portfolio website import | `portfolio.json` |
| Detailed technical write-up | `PORTFOLIO.md` |
| Quick project overview | `README.md` |
| Tech stack list | Both (JSON has structured format) |
| Code examples | `PORTFOLIO.md` |
| Screenshots | Both (same URLs referenced) |
| Live demo link | Both (`https://www.247map.app`) |

## Customization

Feel free to:
- Extract sections from PORTFOLIO.md for blog posts
- Modify portfolio.json structure to fit your website's schema
- Add additional screenshots or metrics
- Update future enhancements as they're completed
- Add links to blog posts or articles about the project

## Questions?

For questions about this portfolio documentation or the project itself, 
visit the [GitHub repository](https://github.com/p-o-f/overnight-map).
