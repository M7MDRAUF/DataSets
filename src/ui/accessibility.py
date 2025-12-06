"""
Accessibility Module for CineMatch V2.1.6

Provides WCAG-compliant accessibility features including
color contrast, screen reader support, and keyboard navigation.

Phase 6 - Task 6.7: Accessibility Features
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class WCAGLevel(Enum):
    """WCAG conformance levels."""
    A = "A"
    AA = "AA"
    AAA = "AAA"


class AccessibilityRole(Enum):
    """ARIA roles for elements."""
    BUTTON = "button"
    LINK = "link"
    NAVIGATION = "navigation"
    MAIN = "main"
    BANNER = "banner"
    CONTENTINFO = "contentinfo"
    COMPLEMENTARY = "complementary"
    SEARCH = "search"
    FORM = "form"
    LIST = "list"
    LISTITEM = "listitem"
    MENU = "menu"
    MENUITEM = "menuitem"
    DIALOG = "dialog"
    ALERT = "alert"
    ALERTDIALOG = "alertdialog"
    PROGRESSBAR = "progressbar"
    TAB = "tab"
    TABLIST = "tablist"
    TABPANEL = "tabpanel"
    TOOLTIP = "tooltip"
    GRID = "grid"
    GRIDCELL = "gridcell"
    ROW = "row"
    COLUMNHEADER = "columnheader"
    ROWHEADER = "rowheader"
    IMG = "img"
    ARTICLE = "article"
    REGION = "region"
    STATUS = "status"
    TIMER = "timer"
    LOG = "log"
    MARQUEE = "marquee"


@dataclass
class ContrastResult:
    """Result of contrast check."""
    ratio: float
    passes_aa: bool
    passes_aa_large: bool
    passes_aaa: bool
    passes_aaa_large: bool
    foreground: str
    background: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ratio": round(self.ratio, 2),
            "passes_aa": self.passes_aa,
            "passes_aa_large": self.passes_aa_large,
            "passes_aaa": self.passes_aaa,
            "passes_aaa_large": self.passes_aaa_large,
            "foreground": self.foreground,
            "background": self.background
        }


class ColorContrastChecker:
    """Check color contrast for WCAG compliance."""
    
    # WCAG contrast ratio requirements
    AA_NORMAL = 4.5
    AA_LARGE = 3.0
    AAA_NORMAL = 7.0
    AAA_LARGE = 4.5
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c*2 for c in hex_color])
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def get_luminance(r: int, g: int, b: int) -> float:
        """Calculate relative luminance (WCAG formula)."""
        def adjust(c: int) -> float:
            c = c / 255
            if c <= 0.03928:
                return c / 12.92
            return ((c + 0.055) / 1.055) ** 2.4
        
        return 0.2126 * adjust(r) + 0.7152 * adjust(g) + 0.0722 * adjust(b)
    
    @classmethod
    def get_contrast_ratio(cls, color1: str, color2: str) -> float:
        """Calculate contrast ratio between two colors."""
        rgb1 = cls.hex_to_rgb(color1)
        rgb2 = cls.hex_to_rgb(color2)
        
        lum1 = cls.get_luminance(*rgb1)
        lum2 = cls.get_luminance(*rgb2)
        
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        
        return (lighter + 0.05) / (darker + 0.05)
    
    @classmethod
    def check_contrast(
        cls,
        foreground: str,
        background: str
    ) -> ContrastResult:
        """Check contrast between foreground and background colors."""
        ratio = cls.get_contrast_ratio(foreground, background)
        
        return ContrastResult(
            ratio=ratio,
            passes_aa=ratio >= cls.AA_NORMAL,
            passes_aa_large=ratio >= cls.AA_LARGE,
            passes_aaa=ratio >= cls.AAA_NORMAL,
            passes_aaa_large=ratio >= cls.AAA_LARGE,
            foreground=foreground,
            background=background
        )
    
    @classmethod
    def suggest_accessible_color(
        cls,
        foreground: str,
        background: str,
        target_level: WCAGLevel = WCAGLevel.AA,
        is_large_text: bool = False
    ) -> Optional[str]:
        """Suggest an accessible alternative color."""
        target_ratio = {
            (WCAGLevel.A, False): cls.AA_NORMAL,
            (WCAGLevel.A, True): cls.AA_LARGE,
            (WCAGLevel.AA, False): cls.AA_NORMAL,
            (WCAGLevel.AA, True): cls.AA_LARGE,
            (WCAGLevel.AAA, False): cls.AAA_NORMAL,
            (WCAGLevel.AAA, True): cls.AAA_LARGE,
        }.get((target_level, is_large_text), cls.AA_NORMAL)
        
        fg_rgb = list(cls.hex_to_rgb(foreground))
        bg_luminance = cls.get_luminance(*cls.hex_to_rgb(background))
        
        # Try adjusting brightness
        for step in range(100):
            # Darken or lighten based on background
            if bg_luminance > 0.5:
                # Dark background, darken foreground
                adjusted = [max(0, c - step * 2) for c in fg_rgb]
            else:
                # Light background, lighten foreground
                adjusted = [min(255, c + step * 2) for c in fg_rgb]
            
            new_color = '#{:02x}{:02x}{:02x}'.format(*adjusted)
            ratio = cls.get_contrast_ratio(new_color, background)
            
            if ratio >= target_ratio:
                return new_color
        
        # Return black or white as fallback
        return "#000000" if bg_luminance > 0.5 else "#ffffff"


@dataclass
class AccessibilityIssue:
    """An accessibility issue found during audit."""
    code: str
    level: WCAGLevel
    criterion: str
    description: str
    element: Optional[str] = None
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "level": self.level.value,
            "criterion": self.criterion,
            "description": self.description,
            "element": self.element,
            "suggestion": self.suggestion
        }


class AccessibilityAuditor:
    """Audit content for accessibility issues."""
    
    @staticmethod
    def audit_html(html: str) -> List[AccessibilityIssue]:
        """Audit HTML content for accessibility issues."""
        issues = []
        
        # Check for images without alt text
        img_pattern = r'<img[^>]*(?!alt=)[^>]*>'
        for match in re.finditer(img_pattern, html, re.IGNORECASE):
            if 'alt=' not in match.group():
                issues.append(AccessibilityIssue(
                    code="IMG_NO_ALT",
                    level=WCAGLevel.A,
                    criterion="1.1.1 Non-text Content",
                    description="Image missing alt attribute",
                    element=match.group()[:100],
                    suggestion="Add descriptive alt text to all images"
                ))
        
        # Check for links without text
        link_pattern = r'<a[^>]*>(\s*)</a>'
        for match in re.finditer(link_pattern, html, re.IGNORECASE):
            issues.append(AccessibilityIssue(
                code="LINK_EMPTY",
                level=WCAGLevel.A,
                criterion="2.4.4 Link Purpose",
                description="Link has no text content",
                element=match.group()[:100],
                suggestion="Add descriptive text to links"
            ))
        
        # Check for form inputs without labels
        input_pattern = r'<input[^>]*(?!aria-label)[^>]*>'
        for match in re.finditer(input_pattern, html, re.IGNORECASE):
            element = match.group()
            if 'type="hidden"' not in element.lower():
                if 'id=' not in element and 'aria-label=' not in element:
                    issues.append(AccessibilityIssue(
                        code="INPUT_NO_LABEL",
                        level=WCAGLevel.A,
                        criterion="1.3.1 Info and Relationships",
                        description="Form input missing label",
                        element=element[:100],
                        suggestion="Add label element or aria-label attribute"
                    ))
        
        # Check for missing language attribute
        if '<html' in html.lower() and 'lang=' not in html.lower()[:500]:
            issues.append(AccessibilityIssue(
                code="HTML_NO_LANG",
                level=WCAGLevel.A,
                criterion="3.1.1 Language of Page",
                description="HTML element missing lang attribute",
                suggestion="Add lang attribute to html element"
            ))
        
        # Check for missing document title
        if '<title>' not in html.lower() or '<title></title>' in html.lower():
            issues.append(AccessibilityIssue(
                code="NO_TITLE",
                level=WCAGLevel.A,
                criterion="2.4.2 Page Titled",
                description="Page missing or empty title",
                suggestion="Add descriptive title to page"
            ))
        
        # Check for heading order
        headings = re.findall(r'<h([1-6])[^>]*>', html, re.IGNORECASE)
        if headings:
            prev_level = 0
            for level in [int(h) for h in headings]:
                if prev_level > 0 and level > prev_level + 1:
                    issues.append(AccessibilityIssue(
                        code="HEADING_SKIP",
                        level=WCAGLevel.A,
                        criterion="1.3.1 Info and Relationships",
                        description=f"Heading level skipped from h{prev_level} to h{level}",
                        suggestion="Use sequential heading levels"
                    ))
                prev_level = level
        
        return issues
    
    @staticmethod
    def audit_text(text: str) -> List[AccessibilityIssue]:
        """Audit text content for accessibility issues."""
        issues = []
        
        # Check for very long paragraphs (readability)
        paragraphs = text.split('\n\n')
        for i, para in enumerate(paragraphs):
            word_count = len(para.split())
            if word_count > 150:
                issues.append(AccessibilityIssue(
                    code="LONG_PARAGRAPH",
                    level=WCAGLevel.AAA,
                    criterion="3.1.5 Reading Level",
                    description=f"Paragraph {i+1} has {word_count} words",
                    suggestion="Break long paragraphs into smaller sections"
                ))
        
        # Check for ALL CAPS text (can be harder to read)
        caps_pattern = r'\b[A-Z]{10,}\b'
        caps_matches = re.findall(caps_pattern, text)
        if caps_matches:
            issues.append(AccessibilityIssue(
                code="ALL_CAPS",
                level=WCAGLevel.AAA,
                criterion="3.1.5 Reading Level",
                description="Text contains extended ALL CAPS sections",
                suggestion="Use sentence case for better readability"
            ))
        
        return issues


@dataclass
class AriaAttributes:
    """ARIA attributes for an element."""
    role: Optional[AccessibilityRole] = None
    label: Optional[str] = None
    labelledby: Optional[str] = None
    describedby: Optional[str] = None
    expanded: Optional[bool] = None
    hidden: Optional[bool] = None
    live: Optional[str] = None  # off, polite, assertive
    current: Optional[str] = None  # page, step, location, date, time, true
    pressed: Optional[bool] = None
    selected: Optional[bool] = None
    checked: Optional[Union[bool, str]] = None  # true, false, mixed
    disabled: Optional[bool] = None
    controls: Optional[str] = None
    owns: Optional[str] = None
    haspopup: Optional[str] = None
    level: Optional[int] = None
    valuemin: Optional[float] = None
    valuemax: Optional[float] = None
    valuenow: Optional[float] = None
    valuetext: Optional[str] = None
    
    def to_attributes(self) -> Dict[str, str]:
        """Convert to HTML attributes dictionary."""
        attrs = {}
        
        if self.role:
            attrs["role"] = self.role.value
        if self.label:
            attrs["aria-label"] = self.label
        if self.labelledby:
            attrs["aria-labelledby"] = self.labelledby
        if self.describedby:
            attrs["aria-describedby"] = self.describedby
        if self.expanded is not None:
            attrs["aria-expanded"] = str(self.expanded).lower()
        if self.hidden is not None:
            attrs["aria-hidden"] = str(self.hidden).lower()
        if self.live:
            attrs["aria-live"] = self.live
        if self.current:
            attrs["aria-current"] = self.current
        if self.pressed is not None:
            attrs["aria-pressed"] = str(self.pressed).lower()
        if self.selected is not None:
            attrs["aria-selected"] = str(self.selected).lower()
        if self.checked is not None:
            if isinstance(self.checked, bool):
                attrs["aria-checked"] = str(self.checked).lower()
            else:
                attrs["aria-checked"] = self.checked
        if self.disabled is not None:
            attrs["aria-disabled"] = str(self.disabled).lower()
        if self.controls:
            attrs["aria-controls"] = self.controls
        if self.owns:
            attrs["aria-owns"] = self.owns
        if self.haspopup:
            attrs["aria-haspopup"] = self.haspopup
        if self.level is not None:
            attrs["aria-level"] = str(self.level)
        if self.valuemin is not None:
            attrs["aria-valuemin"] = str(self.valuemin)
        if self.valuemax is not None:
            attrs["aria-valuemax"] = str(self.valuemax)
        if self.valuenow is not None:
            attrs["aria-valuenow"] = str(self.valuenow)
        if self.valuetext:
            attrs["aria-valuetext"] = self.valuetext
        
        return attrs
    
    def to_html_string(self) -> str:
        """Convert to HTML attribute string."""
        attrs = self.to_attributes()
        return " ".join(f'{k}="{v}"' for k, v in attrs.items())


class AccessibilityHelper:
    """Helper utilities for accessibility."""
    
    @staticmethod
    def create_skip_link(target_id: str, text: str = "Skip to main content") -> str:
        """Create a skip navigation link."""
        return f'''
        <a href="#{target_id}" class="skip-link" 
           style="position: absolute; left: -9999px; z-index: 999; 
                  padding: 1em; background: white; color: black;
                  text-decoration: underline;">
            {text}
        </a>
        <style>
            .skip-link:focus {{
                left: 50%;
                transform: translateX(-50%);
            }}
        </style>
        '''
    
    @staticmethod
    def create_live_region(
        content: str,
        politeness: str = "polite",
        atomic: bool = True
    ) -> str:
        """Create ARIA live region for dynamic updates."""
        return f'''
        <div aria-live="{politeness}" 
             aria-atomic="{str(atomic).lower()}"
             role="status">
            {content}
        </div>
        '''
    
    @staticmethod
    def create_accessible_button(
        text: str,
        on_click: Optional[str] = None,
        aria: Optional[AriaAttributes] = None,
        disabled: bool = False
    ) -> str:
        """Create an accessible button."""
        aria_attrs = aria.to_html_string() if aria else ""
        disabled_attr = 'disabled aria-disabled="true"' if disabled else ""
        onclick = f'onclick="{on_click}"' if on_click else ""
        
        return f'''
        <button type="button" {aria_attrs} {disabled_attr} {onclick}>
            {text}
        </button>
        '''
    
    @staticmethod
    def create_accessible_image(
        src: str,
        alt: str,
        decorative: bool = False
    ) -> str:
        """Create an accessible image."""
        if decorative:
            return f'<img src="{src}" alt="" role="presentation">'
        return f'<img src="{src}" alt="{alt}">'
    
    @staticmethod
    def create_accessible_table(
        headers: List[str],
        rows: List[List[str]],
        caption: Optional[str] = None
    ) -> str:
        """Create an accessible data table."""
        html = '<table role="grid">'
        
        if caption:
            html += f'<caption>{caption}</caption>'
        
        # Header row
        html += '<thead><tr>'
        for header in headers:
            html += f'<th scope="col" role="columnheader">{header}</th>'
        html += '</tr></thead>'
        
        # Data rows
        html += '<tbody>'
        for row in rows:
            html += '<tr role="row">'
            for i, cell in enumerate(row):
                if i == 0:
                    html += f'<th scope="row" role="rowheader">{cell}</th>'
                else:
                    html += f'<td role="gridcell">{cell}</td>'
            html += '</tr>'
        html += '</tbody></table>'
        
        return html
    
    @staticmethod
    def get_focus_styles() -> str:
        """Get CSS for visible focus indicators."""
        return '''
        <style>
            /* Visible focus indicators for keyboard navigation */
            *:focus {
                outline: 2px solid #005fcc;
                outline-offset: 2px;
            }
            
            *:focus:not(:focus-visible) {
                outline: none;
            }
            
            *:focus-visible {
                outline: 2px solid #005fcc;
                outline-offset: 2px;
            }
            
            /* High contrast focus for buttons */
            button:focus-visible,
            [role="button"]:focus-visible {
                outline: 3px solid #005fcc;
                outline-offset: 2px;
                box-shadow: 0 0 0 4px rgba(0, 95, 204, 0.3);
            }
            
            /* Skip link styles */
            .skip-link {
                position: absolute;
                left: -9999px;
                z-index: 9999;
                padding: 1em;
                background: #000;
                color: #fff;
                text-decoration: underline;
            }
            
            .skip-link:focus {
                left: 50%;
                transform: translateX(-50%);
                top: 0;
            }
        </style>
        '''
    
    @staticmethod
    def get_reduced_motion_styles() -> str:
        """Get CSS for reduced motion preference."""
        return '''
        <style>
            @media (prefers-reduced-motion: reduce) {
                *,
                *::before,
                *::after {
                    animation-duration: 0.01ms !important;
                    animation-iteration-count: 1 !important;
                    transition-duration: 0.01ms !important;
                    scroll-behavior: auto !important;
                }
            }
        </style>
        '''
    
    @staticmethod
    def get_high_contrast_styles() -> str:
        """Get CSS for high contrast mode."""
        return '''
        <style>
            @media (prefers-contrast: high) {
                * {
                    border-color: currentColor !important;
                }
                
                button, 
                [role="button"],
                input,
                select,
                textarea {
                    border: 2px solid currentColor !important;
                }
            }
        </style>
        '''


class AccessibilitySettings:
    """User accessibility preferences."""
    
    def __init__(self):
        self.reduce_motion: bool = False
        self.high_contrast: bool = False
        self.large_text: bool = False
        self.text_scale: float = 1.0
        self.screen_reader_mode: bool = False
        self.keyboard_only: bool = False
        self.dyslexia_friendly: bool = False
        self.color_blind_mode: Optional[str] = None  # protanopia, deuteranopia, tritanopia
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reduce_motion": self.reduce_motion,
            "high_contrast": self.high_contrast,
            "large_text": self.large_text,
            "text_scale": self.text_scale,
            "screen_reader_mode": self.screen_reader_mode,
            "keyboard_only": self.keyboard_only,
            "dyslexia_friendly": self.dyslexia_friendly,
            "color_blind_mode": self.color_blind_mode
        }
    
    def generate_css(self) -> str:
        """Generate CSS based on settings."""
        css = ""
        
        if self.reduce_motion:
            css += AccessibilityHelper.get_reduced_motion_styles()
        
        if self.high_contrast:
            css += AccessibilityHelper.get_high_contrast_styles()
        
        if self.large_text or self.text_scale != 1.0:
            scale = max(self.text_scale, 1.25 if self.large_text else 1.0)
            css += f'''
            <style>
                html {{ font-size: {scale * 100}%; }}
            </style>
            '''
        
        if self.dyslexia_friendly:
            css += '''
            <style>
                body {
                    font-family: "OpenDyslexic", "Comic Sans MS", sans-serif;
                    letter-spacing: 0.05em;
                    word-spacing: 0.1em;
                    line-height: 1.8;
                }
            </style>
            '''
        
        return css


class AccessibilityManager:
    """Manages accessibility features."""
    
    def __init__(self):
        self._user_settings: Dict[str, AccessibilitySettings] = {}
        self._default_settings = AccessibilitySettings()
    
    def get_settings(self, user_id: Optional[str] = None) -> AccessibilitySettings:
        """Get accessibility settings for user."""
        if user_id and user_id in self._user_settings:
            return self._user_settings[user_id]
        return self._default_settings
    
    def set_settings(
        self,
        settings: AccessibilitySettings,
        user_id: Optional[str] = None
    ) -> None:
        """Set accessibility settings for user."""
        if user_id:
            self._user_settings[user_id] = settings
        else:
            self._default_settings = settings
    
    def update_settings(
        self,
        user_id: Optional[str] = None,
        **updates: Any
    ) -> AccessibilitySettings:
        """Update specific settings."""
        settings = self.get_settings(user_id)
        for key, value in updates.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        return settings
    
    def check_contrast(
        self,
        foreground: str,
        background: str
    ) -> ContrastResult:
        """Check color contrast."""
        return ColorContrastChecker.check_contrast(foreground, background)
    
    def audit(self, html: str) -> List[AccessibilityIssue]:
        """Audit HTML for accessibility issues."""
        return AccessibilityAuditor.audit_html(html)


# Global accessibility manager
_accessibility_manager: Optional[AccessibilityManager] = None


def get_accessibility_manager() -> AccessibilityManager:
    """Get global accessibility manager."""
    global _accessibility_manager
    if _accessibility_manager is None:
        _accessibility_manager = AccessibilityManager()
    return _accessibility_manager


def check_contrast(foreground: str, background: str) -> ContrastResult:
    """Check color contrast ratio."""
    return ColorContrastChecker.check_contrast(foreground, background)


def audit_accessibility(html: str) -> List[AccessibilityIssue]:
    """Audit HTML for accessibility issues."""
    return AccessibilityAuditor.audit_html(html)


# =============================================================================
# WCAG AUTOMATED VALIDATION (Task 2.5: CI-compatible validation)
# =============================================================================

@dataclass
class WCAGValidationResult:
    """Result of WCAG validation for CI pipelines."""
    passed: bool
    level: WCAGLevel
    total_checks: int
    passed_checks: int
    failed_checks: int
    issues: List[AccessibilityIssue]
    contrast_results: List[ContrastResult]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "level": self.level.value,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "issues": [i.to_dict() for i in self.issues],
            "contrast_results": [c.to_dict() for c in self.contrast_results]
        }


# CineMatch color palette for validation
CINEMATCH_COLOR_PALETTE = {
    # Brand colors
    "primary": "#E50914",
    "primary_dark": "#b20710",
    "secondary": "#667eea",
    "accent": "#764ba2",
    
    # Text colors
    "text_dark": "#000000",
    "text_light": "#ffffff",
    "text_muted": "#666666",
    "text_subtle": "#999999",
    
    # Background colors
    "bg_light": "#ffffff",
    "bg_dark": "#1a1a1a",
    "bg_card": "#f8f9fa",
    "bg_overlay": "rgba(0,0,0,0.5)",
    
    # Status colors
    "success": "#28A745",
    "warning": "#F5A623",
    "error": "#DC3545",
    "info": "#4A90D9",
}


def validate_color_palette(
    palette: Dict[str, str] = None,
    level: WCAGLevel = WCAGLevel.AA
) -> List[ContrastResult]:
    """
    Validate color palette for WCAG compliance.
    
    Args:
        palette: Color palette to validate (uses CINEMATCH_COLOR_PALETTE if None)
        level: Target WCAG level (AA or AAA)
        
    Returns:
        List of contrast check results
    """
    if palette is None:
        palette = CINEMATCH_COLOR_PALETTE
    
    results = []
    
    # Define text/background pairs to check
    pairs_to_check = [
        ("text_dark", "bg_light"),
        ("text_light", "bg_dark"),
        ("text_muted", "bg_light"),
        ("text_dark", "bg_card"),
        ("primary", "bg_light"),
        ("primary", "text_light"),
        ("secondary", "bg_light"),
        ("success", "bg_light"),
        ("warning", "bg_light"),
        ("error", "bg_light"),
        ("info", "bg_light"),
    ]
    
    for fg_key, bg_key in pairs_to_check:
        fg = palette.get(fg_key)
        bg = palette.get(bg_key)
        
        if fg and bg and not fg.startswith("rgba"):
            result = ColorContrastChecker.check_contrast(fg, bg)
            results.append(result)
    
    return results


def validate_wcag_compliance(
    html: str = None,
    palette: Dict[str, str] = None,
    level: WCAGLevel = WCAGLevel.AA
) -> WCAGValidationResult:
    """
    Comprehensive WCAG validation for CI pipelines.
    
    Args:
        html: HTML content to audit (optional)
        palette: Color palette to validate (optional)
        level: Target WCAG level
        
    Returns:
        WCAGValidationResult with pass/fail status
    """
    issues: List[AccessibilityIssue] = []
    contrast_results: List[ContrastResult] = []
    
    # Audit HTML if provided
    if html:
        issues.extend(AccessibilityAuditor.audit_html(html))
    
    # Check color palette
    contrast_results = validate_color_palette(palette, level)
    
    # Count failures based on level
    failed_checks = 0
    passed_checks = 0
    
    # Check issues against target level
    for issue in issues:
        if level == WCAGLevel.A and issue.level == WCAGLevel.A:
            failed_checks += 1
        elif level == WCAGLevel.AA and issue.level in (WCAGLevel.A, WCAGLevel.AA):
            failed_checks += 1
        elif level == WCAGLevel.AAA:
            failed_checks += 1
        else:
            passed_checks += 1
    
    # Check contrast results
    for result in contrast_results:
        if level == WCAGLevel.AAA:
            if not result.passes_aaa:
                failed_checks += 1
            else:
                passed_checks += 1
        else:  # AA or A
            if not result.passes_aa:
                failed_checks += 1
            else:
                passed_checks += 1
    
    total_checks = passed_checks + failed_checks
    
    return WCAGValidationResult(
        passed=failed_checks == 0,
        level=level,
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_checks,
        issues=issues,
        contrast_results=contrast_results
    )


def validate_cinematch_accessibility() -> WCAGValidationResult:
    """
    Validate CineMatch application accessibility.
    
    This function can be called from CI pipelines to ensure
    accessibility compliance before deployment.
    
    Returns:
        WCAGValidationResult with detailed report
    """
    return validate_wcag_compliance(
        palette=CINEMATCH_COLOR_PALETTE,
        level=WCAGLevel.AA
    )


# CLI support for CI/CD pipelines
if __name__ == "__main__":
    import sys
    import json
    
    result = validate_cinematch_accessibility()
    
    print("=" * 60)
    print("CineMatch V2.1.6 - WCAG Accessibility Validation")
    print("=" * 60)
    print(f"Target Level: WCAG {result.level.value}")
    print(f"Total Checks: {result.total_checks}")
    print(f"Passed: {result.passed_checks}")
    print(f"Failed: {result.failed_checks}")
    print("=" * 60)
    
    if result.contrast_results:
        print("\nColor Contrast Results:")
        for cr in result.contrast_results:
            status = "✅" if cr.passes_aa else "❌"
            print(f"  {status} {cr.foreground} on {cr.background}: {cr.ratio:.2f}:1")
    
    if result.issues:
        print("\nAccessibility Issues:")
        for issue in result.issues:
            print(f"  ⚠️ [{issue.level.value}] {issue.code}: {issue.description}")
    
    print("\n" + "=" * 60)
    if result.passed:
        print("✅ WCAG Validation PASSED")
        sys.exit(0)
    else:
        print("❌ WCAG Validation FAILED")
        sys.exit(1)
