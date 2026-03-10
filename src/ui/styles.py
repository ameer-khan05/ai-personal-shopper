"""Custom CSS for the Streamlit app."""

CUSTOM_CSS = """
<style>
/* Main container */
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}

/* Product cards */
.product-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    background: #fafafa;
    transition: box-shadow 0.2s;
}
.product-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.product-card h4 {
    margin: 0 0 8px 0;
    color: #1a1a1a;
}
.product-price {
    font-size: 1.4em;
    font-weight: 700;
    color: #0d6efd;
}
.product-rating {
    color: #f5a623;
    font-size: 0.95em;
}
.product-source {
    color: #888;
    font-size: 0.85em;
}
.product-score {
    background: #e8f5e9;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.85em;
    color: #2e7d32;
    display: inline-block;
}

/* Comparison table */
.comparison-header {
    background: #f0f4f8;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    text-align: center;
}

/* Sidebar preference chips */
.pref-chip {
    display: inline-block;
    background: #e3f2fd;
    border-radius: 16px;
    padding: 4px 12px;
    margin: 2px 4px;
    font-size: 0.85em;
    color: #1565c0;
}
.pref-chip.avoid {
    background: #fce4ec;
    color: #c62828;
}

/* Status indicators */
.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.active {
    background: #4caf50;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
</style>
"""
