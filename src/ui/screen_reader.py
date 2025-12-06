"""
Screen Reader Support Module for CineMatch V2.1.6

Provides comprehensive screen reader support including
ARIA live regions, announcements, and semantic content.

Phase 6 - Task 6.9: Screen Reader Support
"""

import html
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class LiveRegionPoliteness(Enum):
    """ARIA live region politeness levels."""
    OFF = "off"
    POLITE = "polite"
    ASSERTIVE = "assertive"


class AnnouncementPriority(Enum):
    """Priority levels for announcements."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Announcement:
    """An announcement to be read by screen readers."""
    message: str
    priority: AnnouncementPriority = AnnouncementPriority.MEDIUM
    politeness: LiveRegionPoliteness = LiveRegionPoliteness.POLITE
    timestamp: datetime = field(default_factory=datetime.utcnow)
    clear_after_ms: int = 0  # 0 means don't auto-clear
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "priority": self.priority.value,
            "politeness": self.politeness.value,
            "timestamp": self.timestamp.isoformat(),
            "clear_after_ms": self.clear_after_ms
        }


class ScreenReaderAnnouncer:
    """Manages screen reader announcements via ARIA live regions."""
    
    def __init__(self):
        self._announcement_queue: List[Announcement] = []
        self._history: List[Announcement] = []
        self._polite_region_id = "sr-announcer-polite"
        self._assertive_region_id = "sr-announcer-assertive"
    
    def announce(
        self,
        message: str,
        priority: AnnouncementPriority = AnnouncementPriority.MEDIUM,
        clear_after_ms: int = 0
    ) -> Announcement:
        """
        Queue an announcement.
        
        Args:
            message: Text to announce
            priority: Announcement priority
            clear_after_ms: Auto-clear after this many milliseconds
            
        Returns:
            The created announcement
        """
        # Determine politeness based on priority
        if priority in (AnnouncementPriority.HIGH, AnnouncementPriority.CRITICAL):
            politeness = LiveRegionPoliteness.ASSERTIVE
        else:
            politeness = LiveRegionPoliteness.POLITE
        
        announcement = Announcement(
            message=message,
            priority=priority,
            politeness=politeness,
            clear_after_ms=clear_after_ms
        )
        
        self._announcement_queue.append(announcement)
        self._history.append(announcement)
        
        # Trim history
        if len(self._history) > 100:
            self._history = self._history[-100:]
        
        return announcement
    
    def announce_polite(self, message: str) -> Announcement:
        """Queue a polite announcement."""
        return self.announce(message, AnnouncementPriority.MEDIUM)
    
    def announce_assertive(self, message: str) -> Announcement:
        """Queue an assertive announcement."""
        return self.announce(message, AnnouncementPriority.HIGH)
    
    def get_pending_announcements(self) -> List[Announcement]:
        """Get and clear pending announcements."""
        announcements = self._announcement_queue.copy()
        self._announcement_queue.clear()
        return announcements
    
    def get_history(self, limit: int = 20) -> List[Announcement]:
        """Get announcement history."""
        return self._history[-limit:]
    
    def generate_live_regions_html(self) -> str:
        """Generate HTML for live regions."""
        return f'''
        <!-- Screen Reader Live Regions -->
        <div id="{self._polite_region_id}" 
             role="status" 
             aria-live="polite" 
             aria-atomic="true"
             class="sr-only"
             style="position: absolute; left: -9999px; width: 1px; height: 1px; overflow: hidden;">
        </div>
        <div id="{self._assertive_region_id}" 
             role="alert" 
             aria-live="assertive" 
             aria-atomic="true"
             class="sr-only"
             style="position: absolute; left: -9999px; width: 1px; height: 1px; overflow: hidden;">
        </div>
        '''
    
    def generate_announcement_js(self) -> str:
        """Generate JavaScript for announcements."""
        return f'''
        <script>
        window.ScreenReaderAnnouncer = {{
            politeRegionId: '{self._polite_region_id}',
            assertiveRegionId: '{self._assertive_region_id}',
            
            announce: function(message, assertive) {{
                const regionId = assertive ? this.assertiveRegionId : this.politeRegionId;
                const region = document.getElementById(regionId);
                if (region) {{
                    // Clear and set to trigger announcement
                    region.textContent = '';
                    setTimeout(function() {{
                        region.textContent = message;
                    }}, 50);
                }}
            }},
            
            announcePolite: function(message) {{
                this.announce(message, false);
            }},
            
            announceAssertive: function(message) {{
                this.announce(message, true);
            }},
            
            clear: function() {{
                document.getElementById(this.politeRegionId).textContent = '';
                document.getElementById(this.assertiveRegionId).textContent = '';
            }}
        }};
        </script>
        '''


class SemanticContent:
    """Helpers for creating semantically correct content."""
    
    @staticmethod
    def heading(
        level: int,
        text: str,
        id: Optional[str] = None,
        extra_attrs: Optional[Dict[str, str]] = None
    ) -> str:
        """Create a semantic heading."""
        if level < 1 or level > 6:
            level = 2
        
        attrs = []
        if id:
            attrs.append(f'id="{id}"')
        if extra_attrs:
            attrs.extend(f'{k}="{v}"' for k, v in extra_attrs.items())
        
        attr_str = " " + " ".join(attrs) if attrs else ""
        return f'<h{level}{attr_str}>{html.escape(text)}</h{level}>'
    
    @staticmethod
    def landmark_region(
        role: str,
        content: str,
        label: Optional[str] = None,
        labelledby: Optional[str] = None
    ) -> str:
        """Create a landmark region."""
        valid_roles = [
            "banner", "navigation", "main", "complementary",
            "contentinfo", "search", "form", "region"
        ]
        
        if role not in valid_roles:
            role = "region"
        
        attrs = [f'role="{role}"']
        if label:
            attrs.append(f'aria-label="{html.escape(label)}"')
        if labelledby:
            attrs.append(f'aria-labelledby="{labelledby}"')
        
        return f'<div {" ".join(attrs)}>{content}</div>'
    
    @staticmethod
    def navigation(
        items: List[Dict[str, str]],
        label: str = "Main navigation",
        current: Optional[str] = None
    ) -> str:
        """Create semantic navigation."""
        nav_html = f'<nav aria-label="{html.escape(label)}"><ul role="list">'
        
        for item in items:
            href = item.get("href", "#")
            text = item.get("text", "")
            item_id = item.get("id", "")
            
            is_current = current and item_id == current
            current_attr = ' aria-current="page"' if is_current else ""
            
            nav_html += f'''
            <li>
                <a href="{html.escape(href)}"{current_attr}>
                    {html.escape(text)}
                </a>
            </li>
            '''
        
        nav_html += '</ul></nav>'
        return nav_html
    
    @staticmethod
    def description_list(items: List[Dict[str, str]]) -> str:
        """Create a description list."""
        dl_html = '<dl>'
        
        for item in items:
            term = item.get("term", "")
            description = item.get("description", "")
            dl_html += f'''
            <dt>{html.escape(term)}</dt>
            <dd>{html.escape(description)}</dd>
            '''
        
        dl_html += '</dl>'
        return dl_html
    
    @staticmethod
    def figure(
        content: str,
        caption: str,
        caption_id: Optional[str] = None
    ) -> str:
        """Create a figure with caption."""
        cap_id = caption_id or f"fig-caption-{hash(caption) % 10000}"
        return f'''
        <figure aria-labelledby="{cap_id}">
            {content}
            <figcaption id="{cap_id}">{html.escape(caption)}</figcaption>
        </figure>
        '''
    
    @staticmethod
    def movie_card(
        movie: Dict[str, Any],
        rating: Optional[float] = None
    ) -> str:
        """Create accessible movie card."""
        title = movie.get("title", "Unknown")
        year = movie.get("year", "")
        genres = movie.get("genres", [])
        movie_id = movie.get("movie_id", "")
        
        genres_str = ", ".join(genres) if isinstance(genres, list) else str(genres)
        rating_str = f", rated {rating:.1f} stars" if rating else ""
        
        # Create descriptive label
        aria_label = f"{title}"
        if year:
            aria_label += f" ({year})"
        if genres_str:
            aria_label += f", genres: {genres_str}"
        aria_label += rating_str
        
        return f'''
        <article class="movie-card" 
                 role="article"
                 aria-label="{html.escape(aria_label)}">
            <h3 class="movie-title">{html.escape(title)}</h3>
            <p class="movie-year" aria-label="Year">{year}</p>
            <p class="movie-genres" aria-label="Genres">{html.escape(genres_str)}</p>
            {f'<p class="movie-rating" aria-label="Rating">{rating:.1f} stars</p>' if rating else ''}
        </article>
        '''
    
    @staticmethod
    def recommendation_list(
        recommendations: List[Dict[str, Any]],
        list_label: str = "Recommended movies"
    ) -> str:
        """Create accessible recommendation list."""
        list_html = f'''
        <section aria-label="{html.escape(list_label)}">
            <h2>{html.escape(list_label)}</h2>
            <ul role="list" aria-label="{html.escape(list_label)}">
        '''
        
        for i, rec in enumerate(recommendations):
            title = rec.get("title", "Unknown")
            score = rec.get("score", 0)
            rank = i + 1
            
            list_html += f'''
            <li role="listitem" aria-label="Rank {rank}: {html.escape(title)}, match score {score:.0%}">
                <span class="rank" aria-hidden="true">{rank}</span>
                <span class="title">{html.escape(title)}</span>
                <span class="score" aria-label="match score">{score:.0%}</span>
            </li>
            '''
        
        list_html += '</ul></section>'
        return list_html
    
    @staticmethod
    def rating_input(
        movie_id: str,
        movie_title: str,
        current_rating: Optional[float] = None,
        max_rating: float = 5.0
    ) -> str:
        """Create accessible rating input."""
        current_str = f", currently rated {current_rating}" if current_rating else ""
        
        return f'''
        <fieldset class="rating-input" aria-label="Rate {html.escape(movie_title)}">
            <legend class="sr-only">Rate {html.escape(movie_title)}{current_str}</legend>
            <div role="radiogroup" aria-label="Rating">
                {SemanticContent._rating_stars(movie_id, max_rating, current_rating)}
            </div>
        </fieldset>
        '''
    
    @staticmethod
    def _rating_stars(
        movie_id: str,
        max_rating: float,
        current: Optional[float]
    ) -> str:
        """Generate rating star inputs."""
        stars_html = ""
        values = [0.5 * i for i in range(1, int(max_rating * 2) + 1)]
        
        for value in values:
            checked = ' checked' if current and abs(value - current) < 0.01 else ''
            stars_html += f'''
            <input type="radio" 
                   name="rating-{movie_id}" 
                   value="{value}" 
                   id="rating-{movie_id}-{int(value*10)}"
                   aria-label="{value} stars"{checked}>
            <label for="rating-{movie_id}-{int(value*10)}" 
                   aria-hidden="true">★</label>
            '''
        
        return stars_html


class ScreenReaderHelper:
    """Utility class for screen reader support."""
    
    @staticmethod
    def visually_hidden_css() -> str:
        """CSS for visually hidden but screen reader accessible content."""
        return '''
        <style>
            .sr-only {
                position: absolute;
                width: 1px;
                height: 1px;
                padding: 0;
                margin: -1px;
                overflow: hidden;
                clip: rect(0, 0, 0, 0);
                white-space: nowrap;
                border: 0;
            }
            
            .sr-only-focusable:focus,
            .sr-only-focusable:active {
                position: static;
                width: auto;
                height: auto;
                overflow: visible;
                clip: auto;
                white-space: normal;
            }
        </style>
        '''
    
    @staticmethod
    def sr_only(text: str) -> str:
        """Create visually hidden text for screen readers."""
        return f'<span class="sr-only">{html.escape(text)}</span>'
    
    @staticmethod
    def hide_from_sr(content: str) -> str:
        """Hide content from screen readers."""
        return f'<span aria-hidden="true">{content}</span>'
    
    @staticmethod
    def loading_indicator(
        loading_text: str = "Loading...",
        progress: Optional[int] = None
    ) -> str:
        """Create accessible loading indicator."""
        if progress is not None:
            return f'''
            <div role="progressbar" 
                 aria-valuenow="{progress}" 
                 aria-valuemin="0" 
                 aria-valuemax="100"
                 aria-label="{html.escape(loading_text)} {progress}% complete">
                <span class="sr-only">{html.escape(loading_text)} {progress}% complete</span>
            </div>
            '''
        else:
            return f'''
            <div role="status" aria-live="polite" aria-label="{html.escape(loading_text)}">
                <span class="sr-only">{html.escape(loading_text)}</span>
                <span aria-hidden="true" class="spinner"></span>
            </div>
            '''
    
    @staticmethod
    def status_message(
        message: str,
        type: str = "info"  # info, success, warning, error
    ) -> str:
        """Create accessible status message."""
        role = "alert" if type in ("warning", "error") else "status"
        politeness = "assertive" if type in ("warning", "error") else "polite"
        
        return f'''
        <div role="{role}" 
             aria-live="{politeness}"
             class="status-message status-{type}">
            {html.escape(message)}
        </div>
        '''
    
    @staticmethod
    def modal_dialog(
        title: str,
        content: str,
        dialog_id: str,
        description: Optional[str] = None
    ) -> str:
        """Create accessible modal dialog."""
        title_id = f"{dialog_id}-title"
        desc_id = f"{dialog_id}-desc" if description else ""
        
        describedby = f'aria-describedby="{desc_id}"' if desc_id else ""
        
        return f'''
        <div id="{dialog_id}"
             role="dialog"
             aria-modal="true"
             aria-labelledby="{title_id}"
             {describedby}
             class="modal">
            <div class="modal-content">
                <h2 id="{title_id}">{html.escape(title)}</h2>
                {f'<p id="{desc_id}">{html.escape(description)}</p>' if description else ''}
                <div class="modal-body">
                    {content}
                </div>
                <button type="button" 
                        class="modal-close"
                        aria-label="Close dialog">
                    <span aria-hidden="true">×</span>
                </button>
            </div>
        </div>
        '''
    
    @staticmethod
    def tab_panel(
        tabs: List[Dict[str, str]],
        active_tab: str,
        panel_contents: Dict[str, str]
    ) -> str:
        """Create accessible tab panel."""
        tablist_id = f"tablist-{hash(str(tabs)) % 10000}"
        
        tablist_html = f'<div role="tablist" id="{tablist_id}" aria-label="Tabs">'
        panels_html = ""
        
        for tab in tabs:
            tab_id = tab.get("id", "")
            tab_label = tab.get("label", "")
            panel_id = f"panel-{tab_id}"
            is_active = tab_id == active_tab
            
            tablist_html += f'''
            <button role="tab"
                    id="tab-{tab_id}"
                    aria-selected="{str(is_active).lower()}"
                    aria-controls="{panel_id}"
                    tabindex="{0 if is_active else -1}">
                {html.escape(tab_label)}
            </button>
            '''
            
            panels_html += f'''
            <div role="tabpanel"
                 id="{panel_id}"
                 aria-labelledby="tab-{tab_id}"
                 tabindex="0"
                 {'hidden' if not is_active else ''}>
                {panel_contents.get(tab_id, '')}
            </div>
            '''
        
        tablist_html += '</div>'
        
        return tablist_html + panels_html
    
    @staticmethod
    def format_number_for_sr(number: Union[int, float]) -> str:
        """Format number for clear screen reader pronunciation."""
        if isinstance(number, float):
            # Format floats to avoid "point" confusion
            if number == int(number):
                return str(int(number))
            return f"{number:.2f}".rstrip('0').rstrip('.')
        return str(number)
    
    @staticmethod
    def format_rating_for_sr(rating: float, max_rating: float = 5.0) -> str:
        """Format rating for screen readers."""
        return f"{rating:.1f} out of {max_rating:.0f} stars"
    
    @staticmethod
    def format_percentage_for_sr(value: float) -> str:
        """Format percentage for screen readers."""
        return f"{value:.0f} percent"


class ScreenReaderManager:
    """Manages screen reader support features."""
    
    def __init__(self):
        self._announcer = ScreenReaderAnnouncer()
        self._enabled = True
    
    @property
    def announcer(self) -> ScreenReaderAnnouncer:
        """Get the announcer."""
        return self._announcer
    
    def announce(
        self,
        message: str,
        priority: AnnouncementPriority = AnnouncementPriority.MEDIUM
    ) -> Announcement:
        """Make an announcement."""
        return self._announcer.announce(message, priority)
    
    def generate_support_html(self) -> str:
        """Generate all HTML for screen reader support."""
        return (
            ScreenReaderHelper.visually_hidden_css() +
            self._announcer.generate_live_regions_html() +
            self._announcer.generate_announcement_js()
        )
    
    def enable(self) -> None:
        """Enable screen reader support."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable screen reader support."""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """Check if screen reader support is enabled."""
        return self._enabled


# Global instance
_sr_manager: Optional[ScreenReaderManager] = None


def get_screen_reader_manager() -> ScreenReaderManager:
    """Get global screen reader manager."""
    global _sr_manager
    if _sr_manager is None:
        _sr_manager = ScreenReaderManager()
    return _sr_manager


def announce(
    message: str,
    priority: AnnouncementPriority = AnnouncementPriority.MEDIUM
) -> Announcement:
    """Make a screen reader announcement."""
    return get_screen_reader_manager().announce(message, priority)


def sr_only(text: str) -> str:
    """Create visually hidden text for screen readers."""
    return ScreenReaderHelper.sr_only(text)
