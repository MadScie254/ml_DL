"""
Dark Mode Test for Enhanced Dashboard
Simple test to verify dark mode styling works correctly
"""

import streamlit as st

st.set_page_config(
    page_title="Dark Mode Test",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* CSS Variables for adaptive theming */
:root {
    --card-background: #f8f9fa;
    --text-color: #333;
    --border-color: #e9ecef;
}

/* Dark mode detection and overrides */
@media (prefers-color-scheme: dark) {
    :root {
        --card-background: rgba(40, 42, 54, 0.8);
        --text-color: #f8f9fa;
        --border-color: rgba(255, 255, 255, 0.1);
    }
}

/* Streamlit dark theme detection */
[data-theme="dark"] {
    --card-background: rgba(40, 42, 54, 0.8) !important;
    --text-color: #f8f9fa !important;
    --border-color: rgba(255, 255, 255, 0.1) !important;
}

/* Additional fallback for Streamlit dark backgrounds */
.st-emotion-cache-13k62yr {
    --card-background: rgba(40, 42, 54, 0.8) !important;
    --text-color: #f8f9fa !important;
    --border-color: rgba(255, 255, 255, 0.1) !important;
}

.test-card {
    background: var(--card-background, #f8f9fa);
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1rem 0;
    border: 1px solid var(--border-color, #e9ecef);
    color: var(--text-color, #333);
}

.sidebar-section {
    background: var(--card-background, #f8f9fa);
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border: 1px solid var(--border-color, #e9ecef);
    color: var(--text-color, #333);
}

/* Force dark mode styles when detected */
body[class*="dark"] .test-card,
body[class*="dark"] .sidebar-section {
    background: rgba(40, 42, 54, 0.8) !important;
    color: #f8f9fa !important;
    border-color: rgba(255, 255, 255, 0.1) !important;
}
</style>

<script>
// Enhanced dark mode detection
function adaptToTheme() {
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const streamlitTheme = document.body.style.backgroundColor;
    const hasStreamlitDark = streamlitTheme.includes('14, 17, 23') || streamlitTheme.includes('rgb(14, 17, 23)');
    
    if (isDark || hasStreamlitDark) {
        document.documentElement.style.setProperty('--card-background', 'rgba(40, 42, 54, 0.8)');
        document.documentElement.style.setProperty('--text-color', '#f8f9fa');
        document.documentElement.style.setProperty('--border-color', 'rgba(255, 255, 255, 0.1)');
    } else {
        document.documentElement.style.setProperty('--card-background', '#f8f9fa');
        document.documentElement.style.setProperty('--text-color', '#333');
        document.documentElement.style.setProperty('--border-color', '#e9ecef');
    }
}

// Run immediately and on changes
adaptToTheme();
if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addListener(adaptToTheme);
}

// Monitor for Streamlit theme changes
const observer = new MutationObserver(adaptToTheme);
observer.observe(document.body, { 
    attributes: true, 
    attributeFilter: ['style', 'class'] 
});
</script>
""", unsafe_allow_html=True)

st.title("🌙 Dark Mode Test")

# Test cards in main area
st.markdown("""
<div class="test-card">
    <h3>Test Card 1</h3>
    <p>This text should be visible in both light and dark modes.</p>
    <p>Background should adapt automatically.</p>
</div>

<div class="test-card">
    <h3>Test Card 2</h3>
    <p>Another test card to verify styling.</p>
    <p>Text should have good contrast in all themes.</p>
</div>
""", unsafe_allow_html=True)

# Test sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <h4>Sidebar Test</h4>
        <p>This sidebar content should be readable in dark mode.</p>
        <p>Background should adapt to theme.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("Regular sidebar content")
    st.metric("Test Metric", "95.8%", "2.1%")

# Theme information
col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Theme Info")
    st.write("Check if the cards above are readable!")
    
with col2:
    st.subheader("Instructions")
    st.write("1. Switch between light/dark mode in Streamlit")
    st.write("2. Check if all text is visible") 
    st.write("3. Cards should have appropriate backgrounds")

st.success("✅ If you can read this clearly, dark mode is working!")
