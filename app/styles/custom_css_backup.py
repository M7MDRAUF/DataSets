"""
CineMatch V2.1 - Custom CSS Theme System

Netflix-inspired dark theme with genre-based dynamic colors.
Mobile-responsive design with breakpoints.

Author: CineMatch Team
Date: November 11, 2025
"""

# Project Brand Colors - New Color Scheme
PRIMARY_DARK = "#0C2B4E"      # Darkest blue - Main backgrounds
SECONDARY_DARK = "#1A3D64"    # Medium dark blue - Sidebar, cards
ACCENT_BLUE = "#1D546C"       # Accent blue - Highlights, borders
LIGHT_COLOR = "#F4F4F4"       # Light gray - Text, light elements

# Functional Colors (derived from main palette)
SUCCESS_COLOR = "#1D546C"     # Use accent blue for success
WARNING_COLOR = "#1A3D64"     # Use secondary for warnings
ERROR_COLOR = "#0C2B4E"       # Use primary for errors
INFO_COLOR = "#1D546C"        # Use accent blue for info


def get_custom_css() -> str:
    """
    Get complete custom CSS for enhanced UI.
    
    Returns:
        CSS string to inject via st.markdown()
    """
    return f"""
    <style>
    /* ============= GLOBAL DARK MODE ============= */
    
    /* Force dark background on all elements - ULTRA AGGRESSIVE */
    .stApp,
    .stApp > header,
    .stApp > div,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {{
        background-color: {NETFLIX_BLACK} !important;
        background: {NETFLIX_BLACK} !important;
    }}
    
    /* Main content area - ALL nested elements */
    .main,
    .main *,
    .main > div,
    .main > section,
    [data-testid="stMain"],
    [data-testid="stMain"] *,
    section.main {{
        background-color: {NETFLIX_BLACK} !important;
    }}
    
    /* Sidebar dark mode - FORCE on ALL children */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] section {{
        background-color: {NETFLIX_DARK_GRAY} !important;
        background: {NETFLIX_DARK_GRAY} !important;
    }}
    
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarContent"] > div,
    [data-testid="stSidebarContent"] div {{
        background-color: {NETFLIX_DARK_GRAY} !important;
        background: {NETFLIX_DARK_GRAY} !important;
    }}
    
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarUserContent"] > div,
    [data-testid="stSidebarUserContent"] div {{
        background-color: {NETFLIX_DARK_GRAY} !important;
        background: {NETFLIX_DARK_GRAY} !important;
    }}
    
    /* All background colors */
    div[data-testid="stVerticalBlock"],
    div[data-testid="stHorizontalBlock"],
    section[data-testid="stSidebar"] > div {{
        background-color: transparent !important;
    }}
    
    /* Kill all white backgrounds with extreme prejudice */
    [style*="background: white"],
    [style*="background: #fff"],
    [style*="background: #FFF"],
    [style*="background: rgb(255"],
    [style*="background-color: white"],
    [style*="background-color: #fff"],
    [style*="background-color: #FFF"],
    [style*="background-color: rgb(255"] {{
        background-color: {NETFLIX_BLACK} !important;
        background: {NETFLIX_BLACK} !important;
    }}
    
    /* Sidebar white backgrounds specifically */
    [data-testid="stSidebar"] [style*="background"],
    [data-testid="stSidebarContent"] [style*="background"],
    [data-testid="stSidebarUserContent"] [style*="background"] {{
        background-color: {NETFLIX_DARK_GRAY} !important;
        background: {NETFLIX_DARK_GRAY} !important;
    }}
    
    /* Remove default Streamlit padding */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
        background-color: {NETFLIX_BLACK} !important;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {NETFLIX_BLACK};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {NETFLIX_RED};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: #FF0A16;
    }}
    
    
    /* ============= TYPOGRAPHY ============= */
    
    h1 {{
        font-size: 3rem !important;
        font-weight: 700 !important;
        color: white !important;
        margin-bottom: 1rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }}
    
    h2 {{
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: white !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }}
    
    h3 {{
        font-size: 1.5rem !important;
        font-weight: 500 !important;
        color: #CCC !important;
        margin-bottom: 0.75rem !important;
    }}
    
    p {{
        font-size: 1rem !important;
        line-height: 1.6 !important;
        color: #DDD !important;
    }}
    
    
    /* ============= MOVIE CARDS ============= */
    
    .movie-card-enhanced {{
        background: linear-gradient(135deg, {NETFLIX_DARK_GRAY} 0%, {NETFLIX_GRAY} 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }}
    
    .movie-card-enhanced:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(229, 9, 20, 0.3);
        border-color: {NETFLIX_RED};
    }}
    
    .movie-title-large {{
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }}
    
    .movie-year {{
        font-size: 1rem;
        color: {NETFLIX_LIGHT_GRAY};
        margin-left: 0.5rem;
    }}
    
    
    /* ============= GENRE BADGES ============= */
    
    .genre-badge {{
        display: inline-block;
        padding: 0.35rem 0.8rem;
        margin: 0.25rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }}
    
    .genre-badge:hover {{
        transform: scale(1.05);
    }}
    
    .genre-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }}
    
    
    /* ============= RATING VISUALS ============= */
    
    .rating-container {{
        display: flex;
        align-items: center;
        margin: 1rem 0;
        padding: 1rem;
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
    }}
    
    .rating-stars {{
        font-size: 1.5rem;
        letter-spacing: 3px;
        color: #FFD700;
    }}
    
    .rating-number {{
        font-size: 2rem;
        font-weight: bold;
        color: {NETFLIX_RED};
        margin-right: 0.5rem;
    }}
    
    
    /* ============= ALGORITHM SELECTOR ============= */
    
    .algorithm-selector-v2 {{
        background: {NETFLIX_DARK_GRAY};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid {NETFLIX_GRAY};
    }}
    
    .algorithm-option {{
        background: {NETFLIX_GRAY};
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }}
    
    .algorithm-option:hover {{
        background: {NETFLIX_DARK_GRAY};
        border-color: {NETFLIX_RED};
        transform: translateX(5px);
    }}
    
    .algorithm-option.selected {{
        border-color: {NETFLIX_RED};
        background: linear-gradient(135deg, {NETFLIX_RED}22 0%, {NETFLIX_GRAY} 100%);
    }}
    
    
    /* ============= METRIC CARDS ============= */
    
    .metric-card-modern {{
        background: linear-gradient(135deg, {NETFLIX_DARK_GRAY} 0%, {NETFLIX_GRAY} 100%);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }}
    
    .metric-card-modern:hover {{
        transform: scale(1.05);
        box-shadow: 0 8px 16px rgba(229, 9, 20, 0.2);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {NETFLIX_RED};
        margin: 0.5rem 0;
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        color: {NETFLIX_LIGHT_GRAY};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .metric-delta-positive {{
        color: {SUCCESS_GREEN};
        font-size: 0.85rem;
    }}
    
    .metric-delta-negative {{
        color: {ERROR_RED};
        font-size: 0.85rem;
    }}
    
    
    /* ============= RECOMMENDATION GRID ============= */
    
    .recommendation-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }}
    
    @media (max-width: 1200px) {{
        .recommendation-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
    }}
    
    @media (max-width: 768px) {{
        .recommendation-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    
    
    /* ============= BUTTONS ============= */
    
    .stButton > button {{
        background: {NETFLIX_RED};
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }}
    
    .stButton > button:hover {{
        background: #FF0A16;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(229, 9, 20, 0.4);
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
    }}
    
    
    /* ============= SELECT BOXES ============= */
    
    .stSelectbox > div > div {{
        background-color: {NETFLIX_GRAY};
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 4px;
        color: white;
    }}
    
    .stSelectbox > div > div:hover {{
        border-color: {NETFLIX_RED};
    }}
    
    
    /* ============= SLIDERS ============= */
    
    .stSlider > div > div > div > div {{
        background-color: {NETFLIX_RED};
    }}
    
    .stSlider > div > div > div {{
        background-color: {NETFLIX_GRAY};
    }}
    
    
    /* ============= DATAFRAMES ============= */
    
    .dataframe {{
        background-color: {NETFLIX_DARK_GRAY} !important;
        color: white !important;
        border: 1px solid {NETFLIX_GRAY} !important;
        border-radius: 8px !important;
    }}
    
    .dataframe th {{
        background-color: {NETFLIX_RED} !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
    }}
    
    .dataframe td {{
        background-color: {NETFLIX_DARK_GRAY} !important;
        color: white !important;
        padding: 0.75rem !important;
        border-bottom: 1px solid {NETFLIX_GRAY} !important;
    }}
    
    .dataframe tr:hover td {{
        background-color: {NETFLIX_GRAY} !important;
    }}
    
    
    /* ============= LOADING ANIMATIONS ============= */
    
    .loading-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
    }}
    
    .loading-text {{
        color: {NETFLIX_LIGHT_GRAY};
        font-size: 1.2rem;
        margin-top: 1rem;
        animation: pulse 1.5s ease-in-out infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    
    
    /* ============= TABS ============= */
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1rem;
        background-color: {NETFLIX_BLACK};
        border-radius: 8px;
        padding: 0.5rem;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {NETFLIX_GRAY};
        color: white;
        border-radius: 4px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border: none;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {NETFLIX_RED};
    }}
    
    
    /* ============= EXPANDERS ============= */
    
    .streamlit-expanderHeader {{
        background-color: {NETFLIX_GRAY};
        color: white;
        border-radius: 4px;
        font-weight: 600;
    }}
    
    .streamlit-expanderHeader:hover {{
        background-color: {NETFLIX_DARK_GRAY};
        border-color: {NETFLIX_RED};
    }}
    
    .streamlit-expanderContent {{
        background-color: {NETFLIX_DARK_GRAY};
        border: 1px solid {NETFLIX_GRAY};
        border-radius: 0 0 4px 4px;
    }}
    
    
    /* ============= ALERTS ============= */
    
    .alert-success {{
        background-color: {SUCCESS_GREEN}22;
        border-left: 4px solid {SUCCESS_GREEN};
        padding: 1rem;
        border-radius: 4px;
        color: white;
        margin: 1rem 0;
    }}
    
    .alert-warning {{
        background-color: {WARNING_YELLOW}22;
        border-left: 4px solid {WARNING_YELLOW};
        padding: 1rem;
        border-radius: 4px;
        color: white;
        margin: 1rem 0;
    }}
    
    .alert-error {{
        background-color: {ERROR_RED}22;
        border-left: 4px solid {ERROR_RED};
        padding: 1rem;
        border-radius: 4px;
        color: white;
        margin: 1rem 0;
    }}
    
    .alert-info {{
        background-color: {INFO_BLUE}22;
        border-left: 4px solid {INFO_BLUE};
        padding: 1rem;
        border-radius: 4px;
        color: white;
        margin: 1rem 0;
    }}
    
    
    /* ============= COMPREHENSIVE DARK MODE FIXES ============= */
    
    /* Override ALL Streamlit backgrounds - Most aggressive approach */
    * {{
        scrollbar-color: {NETFLIX_RED} {NETFLIX_BLACK} !important;
    }}
    
    /* Root level backgrounds */
    html, body, #root, .stApp {{
        background-color: {NETFLIX_BLACK} !important;
    }}
    
    /* Target specific emotion-cache classes that Streamlit uses */
    .st-emotion-cache-1csi29a,
    .st-emotion-cache-10p9htt,
    .st-emotion-cache-11ukie,
    .st-emotion-cache-qmp9ai,
    .st-emotion-cache-1r1cntt,
    .st-emotion-cache-8atqhb,
    .st-emotion-cache-tn0cau,
    .st-emotion-cache-1vo6xi6,
    .st-emotion-cache-1fwbbrh,
    .st-emotion-cache-1s2v671,
    .st-emotion-cache-uujwi5,
    .st-emotion-cache-7e7wz2,
    [class^="st-emotion-cache-"],
    [class*=" st-emotion-cache-"] {{
        background-color: transparent !important;
    }}
    
    /* Sidebar emotion-cache classes */
    [data-testid="stSidebar"] [class^="st-emotion-cache-"],
    [data-testid="stSidebar"] [class*=" st-emotion-cache-"] {{
        background-color: {NETFLIX_DARK_GRAY} !important;
    }}
    
    /* All emotion-cache classes (Streamlit's CSS-in-JS) */
    div[class*="st-emotion-cache"],
    div[class*="element-container"],
    div[class*="stMarkdown"],
    div[class*="stSelectbox"],
    div[class*="stMultiSelect"],
    div[class*="stNumberInput"],
    div[class*="stTextInput"],
    section[class*="st-emotion-cache"],
    div[class*="Block"],
    div[class*="stVertical"],
    div[class*="stHorizontal"] {{
        background-color: transparent !important;
    }}
    
    /* Force sidebar dark mode on ALL nested elements */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] *,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarContent"] *,
    [data-testid="stSidebarHeader"],
    [data-testid="stSidebarHeader"] *,
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarUserContent"] * {{
        background-color: {NETFLIX_DARK_GRAY} !important;
    }}
    
    /* Override white backgrounds specifically */
    div[style*="background-color: white"],
    div[style*="background-color: rgb(255, 255, 255)"],
    div[style*="background: white"],
    div[style*="background: rgb(255, 255, 255)"] {{
        background-color: {NETFLIX_BLACK} !important;
        background: {NETFLIX_BLACK} !important;
    }}
    
    /* Sidebar specific white override */
    [data-testid="stSidebar"] div[style*="background"],
    [data-testid="stSidebarContent"] div[style*="background"] {{
        background-color: {NETFLIX_DARK_GRAY} !important;
        background: {NETFLIX_DARK_GRAY} !important;
    }}
    
    /* Input fields dark mode */
    input[type="text"],
    input[type="number"],
    input[class*="st-"],
    textarea,
    select {{
        background-color: {NETFLIX_GRAY} !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }}
    
    /* Selectbox dropdown - all nested divs */
    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div,
    [class*="st-ae"],
    [class*="st-af"],
    [class*="st-ag"] {{
        background-color: {NETFLIX_GRAY} !important;
        border-color: rgba(255,255,255,0.2) !important;
    }}
    
    /* Dropdown menu items */
    ul[role="listbox"],
    ul[role="listbox"] li {{
        background-color: {NETFLIX_GRAY} !important;
        color: white !important;
    }}
    
    ul[role="listbox"] li:hover {{
        background-color: {NETFLIX_RED} !important;
    }}
    
    /* Text color preservation */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] span {{
        color: inherit !important;
    }}
    
    /* Algorithm card backgrounds */
    .algorithm-card {{
        background-color: {NETFLIX_GRAY} !important;
        border: 1px solid rgba(229, 9, 20, 0.3) !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
    }}
    
    .algorithm-card h3 {{
        color: {NETFLIX_RED} !important;
        margin-bottom: 0.75rem !important;
    }}
    
    .algorithm-card p {{
        color: #DDD !important;
        margin: 0.5rem 0 !important;
    }}
    
    .algorithm-card strong {{
        color: white !important;
    }}
    
    /* Widget labels */
    label[data-testid="stWidgetLabel"],
    label[data-testid="stWidgetLabel"] *,
    label[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] div {{
        color: white !important;
        background-color: transparent !important;
    }}
    
    /* Tooltips */
    div[data-testid="stTooltipIcon"],
    div[data-testid="stTooltipIcon"] * {{
        color: {NETFLIX_LIGHT_GRAY} !important;
    }}
    
    /* Headers everywhere */
    h1, h2, h3, h4, h5, h6 {{
        background-color: transparent !important;
    }}
    
    /* Sidebar headers */
    [data-testid="stSidebarUserContent"] h1,
    [data-testid="stSidebarUserContent"] h2,
    [data-testid="stSidebarUserContent"] h3,
    [data-testid="stSidebarUserContent"] h4,
    [data-testid="stSidebarUserContent"] h5,
    [data-testid="stSidebarUserContent"] h6 {{
        color: white !important;
        background-color: transparent !important;
    }}
    
    /* Number input spinners */
    input[type="number"]::-webkit-inner-spin-button,
    input[type="number"]::-webkit-outer-spin-button {{
        background-color: {NETFLIX_GRAY} !important;
    }}
    
    /* Metrics and stats */
    [data-testid="stMetric"],
    [data-testid="stMetric"] * {{
        background-color: transparent !important;
    }}
    
    /* Charts and visualizations */
    [data-testid="stPlotlyChart"],
    [data-testid="stVegaLiteChart"] {{
        background-color: transparent !important;
    }}
    
    
    /* ============= MOBILE RESPONSIVE ============= */
    
    @media (max-width: 768px) {{
        h1 {{
            font-size: 2rem !important;
        }}
        
        h2 {{
            font-size: 1.5rem !important;
        }}
        
        .movie-title-large {{
            font-size: 1.5rem;
        }}
        
        .metric-value {{
            font-size: 2rem;
        }}
        
        .main .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
        
        .movie-card-enhanced {{
            padding: 1rem;
        }}
    }}
    
    @media (max-width: 480px) {{
        h1 {{
            font-size: 1.5rem !important;
        }}
        
        .movie-title-large {{
            font-size: 1.2rem;
        }}
        
        .genre-badge {{
            font-size: 0.75rem;
            padding: 0.25rem 0.6rem;
        }}
    }}
    
    </style>
    """


def get_hero_section_css() -> str:
    """
    CSS for hero/header sections with animated backgrounds.
    
    Returns:
        CSS string for hero sections
    """
    return """
    <style>
    .hero-section {
        background: linear-gradient(135deg, #141414 0%, #E50914 100%);
        padding: 3rem 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #DDD;
        position: relative;
        z-index: 1;
    }
    </style>
    """
