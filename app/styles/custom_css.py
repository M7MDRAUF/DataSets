"""
CineMatch V2.1 - Custom CSS Theme System

Netflix-inspired dark theme with genre-based dynamic colors.
Mobile-responsive design with breakpoints.

Author: CineMatch Team
Date: November 11, 2025
"""

# Netflix Brand Colors
NETFLIX_RED = "#E50914"
NETFLIX_BLACK = "#141414"
NETFLIX_DARK_GRAY = "#222222"
NETFLIX_GRAY = "#333333"
NETFLIX_LIGHT_GRAY = "#757575"

# Accent Colors
SUCCESS_GREEN = "#4CAF50"
WARNING_YELLOW = "#FFC107"
ERROR_RED = "#F44336"
INFO_BLUE = "#2196F3"


def get_custom_css() -> str:
    """
    Get complete custom CSS for enhanced UI.
    
    Returns:
        CSS string to inject via st.markdown()
    """
    return f"""
    <style>
    /* ============= GLOBAL STYLES ============= */
    
    /* Remove default Streamlit padding */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
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
