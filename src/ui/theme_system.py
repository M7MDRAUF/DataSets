"""
Theme and Customization Module for CineMatch V2.1.6

Provides theme management, UI customization, and styling
capabilities for the Streamlit-based interface.

Phase 6 - Task 6.5: Theme Support
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ColorPalette:
    """Color palette for a theme."""
    primary: str = "#3498db"
    secondary: str = "#2ecc71"
    background: str = "#ffffff"
    surface: str = "#f8f9fa"
    text: str = "#2c3e50"
    text_secondary: str = "#7f8c8d"
    accent: str = "#e74c3c"
    success: str = "#27ae60"
    warning: str = "#f39c12"
    error: str = "#e74c3c"
    info: str = "#3498db"
    border: str = "#dee2e6"
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "background": self.background,
            "surface": self.surface,
            "text": self.text,
            "text_secondary": self.text_secondary,
            "accent": self.accent,
            "success": self.success,
            "warning": self.warning,
            "error": self.error,
            "info": self.info,
            "border": self.border
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'ColorPalette':
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class Typography:
    """Typography settings for a theme."""
    font_family: str = "'Inter', 'Segoe UI', Roboto, sans-serif"
    font_family_mono: str = "'Fira Code', 'Consolas', monospace"
    font_size_xs: str = "0.75rem"
    font_size_sm: str = "0.875rem"
    font_size_base: str = "1rem"
    font_size_lg: str = "1.125rem"
    font_size_xl: str = "1.25rem"
    font_size_2xl: str = "1.5rem"
    font_size_3xl: str = "2rem"
    font_weight_normal: int = 400
    font_weight_medium: int = 500
    font_weight_bold: int = 700
    line_height: float = 1.6
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "font_family": self.font_family,
            "font_family_mono": self.font_family_mono,
            "font_size_xs": self.font_size_xs,
            "font_size_sm": self.font_size_sm,
            "font_size_base": self.font_size_base,
            "font_size_lg": self.font_size_lg,
            "font_size_xl": self.font_size_xl,
            "font_size_2xl": self.font_size_2xl,
            "font_size_3xl": self.font_size_3xl,
            "font_weight_normal": self.font_weight_normal,
            "font_weight_medium": self.font_weight_medium,
            "font_weight_bold": self.font_weight_bold,
            "line_height": self.line_height
        }


@dataclass
class Spacing:
    """Spacing values for a theme."""
    xs: str = "0.25rem"
    sm: str = "0.5rem"
    md: str = "1rem"
    lg: str = "1.5rem"
    xl: str = "2rem"
    xxl: str = "3rem"
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "xs": self.xs,
            "sm": self.sm,
            "md": self.md,
            "lg": self.lg,
            "xl": self.xl,
            "xxl": self.xxl
        }


@dataclass
class BorderRadius:
    """Border radius values for a theme."""
    sm: str = "0.25rem"
    md: str = "0.5rem"
    lg: str = "1rem"
    xl: str = "1.5rem"
    full: str = "9999px"
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "sm": self.sm,
            "md": self.md,
            "lg": self.lg,
            "xl": self.xl,
            "full": self.full
        }


@dataclass
class Shadows:
    """Shadow values for a theme."""
    sm: str = "0 1px 2px rgba(0, 0, 0, 0.05)"
    md: str = "0 4px 6px rgba(0, 0, 0, 0.1)"
    lg: str = "0 10px 15px rgba(0, 0, 0, 0.1)"
    xl: str = "0 20px 25px rgba(0, 0, 0, 0.15)"
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "sm": self.sm,
            "md": self.md,
            "lg": self.lg,
            "xl": self.xl
        }


@dataclass
class Theme:
    """Complete theme definition."""
    name: str
    display_name: str
    description: str = ""
    is_dark: bool = False
    colors: ColorPalette = field(default_factory=ColorPalette)
    typography: Typography = field(default_factory=Typography)
    spacing: Spacing = field(default_factory=Spacing)
    border_radius: BorderRadius = field(default_factory=BorderRadius)
    shadows: Shadows = field(default_factory=Shadows)
    custom_css: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "is_dark": self.is_dark,
            "colors": self.colors.to_dict(),
            "typography": self.typography.to_dict(),
            "spacing": self.spacing.to_dict(),
            "border_radius": self.border_radius.to_dict(),
            "shadows": self.shadows.to_dict(),
            "custom_css": self.custom_css
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Theme':
        return cls(
            name=data["name"],
            display_name=data["display_name"],
            description=data.get("description", ""),
            is_dark=data.get("is_dark", False),
            colors=ColorPalette.from_dict(data.get("colors", {})),
            typography=Typography(**{k: v for k, v in data.get("typography", {}).items() if hasattr(Typography, k)}),
            spacing=Spacing(**{k: v for k, v in data.get("spacing", {}).items() if hasattr(Spacing, k)}),
            border_radius=BorderRadius(**{k: v for k, v in data.get("border_radius", {}).items() if hasattr(BorderRadius, k)}),
            shadows=Shadows(**{k: v for k, v in data.get("shadows", {}).items() if hasattr(Shadows, k)}),
            custom_css=data.get("custom_css", "")
        )
    
    def generate_css(self) -> str:
        """Generate CSS from theme settings."""
        css = f"""
        :root {{
            /* Colors */
            --color-primary: {self.colors.primary};
            --color-secondary: {self.colors.secondary};
            --color-background: {self.colors.background};
            --color-surface: {self.colors.surface};
            --color-text: {self.colors.text};
            --color-text-secondary: {self.colors.text_secondary};
            --color-accent: {self.colors.accent};
            --color-success: {self.colors.success};
            --color-warning: {self.colors.warning};
            --color-error: {self.colors.error};
            --color-info: {self.colors.info};
            --color-border: {self.colors.border};
            
            /* Typography */
            --font-family: {self.typography.font_family};
            --font-family-mono: {self.typography.font_family_mono};
            --font-size-xs: {self.typography.font_size_xs};
            --font-size-sm: {self.typography.font_size_sm};
            --font-size-base: {self.typography.font_size_base};
            --font-size-lg: {self.typography.font_size_lg};
            --font-size-xl: {self.typography.font_size_xl};
            --font-size-2xl: {self.typography.font_size_2xl};
            --font-size-3xl: {self.typography.font_size_3xl};
            --line-height: {self.typography.line_height};
            
            /* Spacing */
            --spacing-xs: {self.spacing.xs};
            --spacing-sm: {self.spacing.sm};
            --spacing-md: {self.spacing.md};
            --spacing-lg: {self.spacing.lg};
            --spacing-xl: {self.spacing.xl};
            --spacing-xxl: {self.spacing.xxl};
            
            /* Border Radius */
            --radius-sm: {self.border_radius.sm};
            --radius-md: {self.border_radius.md};
            --radius-lg: {self.border_radius.lg};
            --radius-xl: {self.border_radius.xl};
            --radius-full: {self.border_radius.full};
            
            /* Shadows */
            --shadow-sm: {self.shadows.sm};
            --shadow-md: {self.shadows.md};
            --shadow-lg: {self.shadows.lg};
            --shadow-xl: {self.shadows.xl};
        }}
        
        /* Base styles */
        body {{
            font-family: var(--font-family);
            font-size: var(--font-size-base);
            line-height: var(--line-height);
            color: var(--color-text);
            background-color: var(--color-background);
        }}
        
        /* Streamlit specific overrides */
        .stApp {{
            background-color: var(--color-background);
        }}
        
        .stButton > button {{
            background-color: var(--color-primary);
            color: white;
            border-radius: var(--radius-md);
            padding: var(--spacing-sm) var(--spacing-md);
            border: none;
            font-weight: {self.typography.font_weight_medium};
            transition: all 0.2s ease;
        }}
        
        .stButton > button:hover {{
            opacity: 0.9;
            box-shadow: var(--shadow-md);
        }}
        
        .stTextInput > div > div > input {{
            border-radius: var(--radius-md);
            border: 1px solid var(--color-border);
            padding: var(--spacing-sm);
        }}
        
        .stSelectbox > div > div {{
            border-radius: var(--radius-md);
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: var(--color-text);
            font-weight: {self.typography.font_weight_bold};
        }}
        
        a {{
            color: var(--color-primary);
        }}
        
        .stMetric {{
            background-color: var(--color-surface);
            border-radius: var(--radius-lg);
            padding: var(--spacing-md);
        }}
        
        .stAlert {{
            border-radius: var(--radius-md);
        }}
        
        /* Card-like containers */
        .css-1r6slb0, .css-keje6w {{
            background-color: var(--color-surface);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-sm);
        }}
        """
        
        if self.custom_css:
            css += f"\n/* Custom CSS */\n{self.custom_css}"
        
        return css
    
    def generate_streamlit_config(self) -> Dict[str, Any]:
        """Generate Streamlit config.toml settings."""
        return {
            "theme": {
                "primaryColor": self.colors.primary,
                "backgroundColor": self.colors.background,
                "secondaryBackgroundColor": self.colors.surface,
                "textColor": self.colors.text,
                "font": self.typography.font_family.split(",")[0].strip("'\"")
            }
        }


# Pre-defined themes

class BuiltinThemes:
    """Built-in theme definitions."""
    
    LIGHT = Theme(
        name="light",
        display_name="Light",
        description="Clean, light theme for comfortable viewing",
        is_dark=False,
        colors=ColorPalette(
            primary="#3498db",
            secondary="#2ecc71",
            background="#ffffff",
            surface="#f8f9fa",
            text="#2c3e50",
            text_secondary="#7f8c8d"
        )
    )
    
    DARK = Theme(
        name="dark",
        display_name="Dark",
        description="Dark theme for reduced eye strain",
        is_dark=True,
        colors=ColorPalette(
            primary="#5dade2",
            secondary="#58d68d",
            background="#1a1a2e",
            surface="#16213e",
            text="#eaeaea",
            text_secondary="#b0b0b0",
            border="#2d2d44"
        ),
        shadows=Shadows(
            sm="0 1px 2px rgba(0, 0, 0, 0.3)",
            md="0 4px 6px rgba(0, 0, 0, 0.4)",
            lg="0 10px 15px rgba(0, 0, 0, 0.4)",
            xl="0 20px 25px rgba(0, 0, 0, 0.5)"
        )
    )
    
    CINEMA = Theme(
        name="cinema",
        display_name="Cinema",
        description="Movie theater inspired dark theme",
        is_dark=True,
        colors=ColorPalette(
            primary="#e50914",  # Netflix red
            secondary="#f5c518",  # IMDb yellow
            background="#141414",
            surface="#1f1f1f",
            text="#ffffff",
            text_secondary="#b3b3b3",
            accent="#e50914",
            border="#333333"
        )
    )
    
    OCEAN = Theme(
        name="ocean",
        display_name="Ocean",
        description="Calming blue ocean theme",
        is_dark=False,
        colors=ColorPalette(
            primary="#0077b6",
            secondary="#00b4d8",
            background="#caf0f8",
            surface="#ade8f4",
            text="#03045e",
            text_secondary="#0077b6",
            accent="#023e8a"
        )
    )
    
    FOREST = Theme(
        name="forest",
        display_name="Forest",
        description="Nature-inspired green theme",
        is_dark=False,
        colors=ColorPalette(
            primary="#2d6a4f",
            secondary="#40916c",
            background="#d8f3dc",
            surface="#b7e4c7",
            text="#1b4332",
            text_secondary="#40916c",
            accent="#081c15"
        )
    )
    
    SUNSET = Theme(
        name="sunset",
        display_name="Sunset",
        description="Warm sunset colors theme",
        is_dark=False,
        colors=ColorPalette(
            primary="#e76f51",
            secondary="#f4a261",
            background="#fefae0",
            surface="#faedcd",
            text="#264653",
            text_secondary="#2a9d8f",
            accent="#e9c46a"
        )
    )
    
    HIGH_CONTRAST = Theme(
        name="high_contrast",
        display_name="High Contrast",
        description="High contrast theme for accessibility",
        is_dark=True,
        colors=ColorPalette(
            primary="#ffff00",
            secondary="#00ffff",
            background="#000000",
            surface="#1a1a1a",
            text="#ffffff",
            text_secondary="#ffffff",
            accent="#ff00ff",
            border="#ffffff"
        )
    )
    
    @classmethod
    def get_all(cls) -> List[Theme]:
        """Get all built-in themes."""
        return [
            cls.LIGHT,
            cls.DARK,
            cls.CINEMA,
            cls.OCEAN,
            cls.FOREST,
            cls.SUNSET,
            cls.HIGH_CONTRAST
        ]
    
    @classmethod
    def get_by_name(cls, name: str) -> Optional[Theme]:
        """Get theme by name."""
        for theme in cls.get_all():
            if theme.name == name:
                return theme
        return None


class ThemeManager:
    """Manages themes and customization."""
    
    def __init__(self, themes_dir: Optional[Path] = None):
        self._themes: Dict[str, Theme] = {}
        self._current_theme: Optional[Theme] = None
        self._themes_dir = themes_dir
        self._change_callbacks: List[Callable[[Theme], None]] = []
        
        # Load built-in themes
        for theme in BuiltinThemes.get_all():
            self._themes[theme.name] = theme
        
        # Load custom themes from directory
        if themes_dir and themes_dir.exists():
            self._load_themes_from_directory(themes_dir)
        
        # Set default theme
        self._current_theme = self._themes.get("light")
    
    def _load_themes_from_directory(self, directory: Path) -> None:
        """Load themes from JSON files in directory."""
        for path in directory.glob("*.json"):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                theme = Theme.from_dict(data)
                self._themes[theme.name] = theme
                logger.info(f"Loaded theme: {theme.name}")
            except Exception as e:
                logger.error(f"Failed to load theme from {path}: {e}")
    
    def register_theme(self, theme: Theme) -> None:
        """Register a custom theme."""
        self._themes[theme.name] = theme
        logger.info(f"Registered theme: {theme.name}")
    
    def unregister_theme(self, name: str) -> bool:
        """Unregister a theme."""
        if name in self._themes:
            del self._themes[name]
            return True
        return False
    
    def get_theme(self, name: str) -> Optional[Theme]:
        """Get theme by name."""
        return self._themes.get(name)
    
    def list_themes(self) -> List[Dict[str, str]]:
        """List all available themes."""
        return [
            {
                "name": theme.name,
                "display_name": theme.display_name,
                "description": theme.description,
                "is_dark": theme.is_dark
            }
            for theme in self._themes.values()
        ]
    
    def set_current_theme(self, name: str) -> bool:
        """Set the current theme."""
        theme = self._themes.get(name)
        if theme:
            self._current_theme = theme
            for callback in self._change_callbacks:
                try:
                    callback(theme)
                except Exception as e:
                    logger.error(f"Theme change callback error: {e}")
            return True
        return False
    
    def get_current_theme(self) -> Optional[Theme]:
        """Get the current theme."""
        return self._current_theme
    
    def on_theme_change(self, callback: Callable[[Theme], None]) -> None:
        """Register callback for theme changes."""
        self._change_callbacks.append(callback)
    
    def save_theme(self, theme: Theme, path: Optional[Path] = None) -> bool:
        """Save theme to JSON file."""
        try:
            if path is None and self._themes_dir:
                path = self._themes_dir / f"{theme.name}.json"
            
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    json.dump(theme.to_dict(), f, indent=2)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to save theme: {e}")
            return False
    
    def create_custom_theme(
        self,
        name: str,
        base_theme: str = "light",
        **customizations: Any
    ) -> Theme:
        """Create a custom theme based on an existing theme."""
        base = self._themes.get(base_theme, BuiltinThemes.LIGHT)
        
        # Create new theme with customizations
        theme_dict = base.to_dict()
        theme_dict["name"] = name
        theme_dict["display_name"] = customizations.get("display_name", name.title())
        theme_dict["description"] = customizations.get("description", f"Custom theme based on {base_theme}")
        
        # Apply color customizations
        if "colors" in customizations:
            theme_dict["colors"].update(customizations["colors"])
        
        # Apply typography customizations
        if "typography" in customizations:
            theme_dict["typography"].update(customizations["typography"])
        
        # Apply custom CSS
        if "custom_css" in customizations:
            theme_dict["custom_css"] = customizations["custom_css"]
        
        theme = Theme.from_dict(theme_dict)
        self.register_theme(theme)
        return theme


class StreamlitThemeApplier:
    """Apply themes to Streamlit application."""
    
    @staticmethod
    def apply_theme(theme: Theme) -> str:
        """
        Generate Streamlit custom CSS to apply theme.
        
        Returns CSS string to inject via st.markdown.
        """
        css = theme.generate_css()
        return f"<style>{css}</style>"
    
    @staticmethod
    def inject_theme(theme: Theme) -> None:
        """Inject theme CSS into Streamlit app."""
        try:
            import streamlit as st
            css = StreamlitThemeApplier.apply_theme(theme)
            st.markdown(css, unsafe_allow_html=True)
        except ImportError:
            logger.warning("Streamlit not available for theme injection")
    
    @staticmethod
    def create_theme_selector(
        theme_manager: ThemeManager,
        key: str = "theme_selector"
    ) -> Optional[Theme]:
        """Create Streamlit theme selector widget."""
        try:
            import streamlit as st
            
            themes = theme_manager.list_themes()
            current = theme_manager.get_current_theme()
            
            theme_names = [t["name"] for t in themes]
            theme_display = [t["display_name"] for t in themes]
            
            current_index = theme_names.index(current.name) if current else 0
            
            selected = st.selectbox(
                "Theme",
                options=theme_names,
                format_func=lambda x: theme_display[theme_names.index(x)],
                index=current_index,
                key=key
            )
            
            if selected and selected != (current.name if current else None):
                theme_manager.set_current_theme(selected)
            
            return theme_manager.get_current_theme()
            
        except ImportError:
            logger.warning("Streamlit not available for theme selector")
            return None


@dataclass
class UICustomization:
    """UI customization settings."""
    show_sidebar: bool = True
    sidebar_width: str = "300px"
    show_header: bool = True
    show_footer: bool = True
    show_logo: bool = True
    logo_url: Optional[str] = None
    compact_mode: bool = False
    animations_enabled: bool = True
    card_layout: str = "grid"  # grid, list
    items_per_page: int = 20
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "show_sidebar": self.show_sidebar,
            "sidebar_width": self.sidebar_width,
            "show_header": self.show_header,
            "show_footer": self.show_footer,
            "show_logo": self.show_logo,
            "logo_url": self.logo_url,
            "compact_mode": self.compact_mode,
            "animations_enabled": self.animations_enabled,
            "card_layout": self.card_layout,
            "items_per_page": self.items_per_page
        }


class CustomizationManager:
    """Manages UI customization settings."""
    
    def __init__(self):
        self._customization = UICustomization()
        self._user_customizations: Dict[str, UICustomization] = {}
    
    def get_customization(self, user_id: Optional[str] = None) -> UICustomization:
        """Get customization for user or default."""
        if user_id and user_id in self._user_customizations:
            return self._user_customizations[user_id]
        return self._customization
    
    def set_customization(
        self,
        customization: UICustomization,
        user_id: Optional[str] = None
    ) -> None:
        """Set customization for user or default."""
        if user_id:
            self._user_customizations[user_id] = customization
        else:
            self._customization = customization
    
    def update_customization(
        self,
        user_id: Optional[str] = None,
        **updates: Any
    ) -> UICustomization:
        """Update specific customization settings."""
        current = self.get_customization(user_id)
        
        for key, value in updates.items():
            if hasattr(current, key):
                setattr(current, key, value)
        
        return current
    
    def reset_customization(self, user_id: Optional[str] = None) -> None:
        """Reset customization to defaults."""
        if user_id and user_id in self._user_customizations:
            del self._user_customizations[user_id]
        else:
            self._customization = UICustomization()


# Global instances
_theme_manager: Optional[ThemeManager] = None
_customization_manager: Optional[CustomizationManager] = None


def get_theme_manager(themes_dir: Optional[Path] = None) -> ThemeManager:
    """Get global theme manager instance."""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager(themes_dir)
    return _theme_manager


def get_customization_manager() -> CustomizationManager:
    """Get global customization manager instance."""
    global _customization_manager
    if _customization_manager is None:
        _customization_manager = CustomizationManager()
    return _customization_manager


def apply_current_theme() -> None:
    """Apply the current theme to Streamlit app."""
    manager = get_theme_manager()
    theme = manager.get_current_theme()
    if theme:
        StreamlitThemeApplier.inject_theme(theme)
