"""
Keyboard Navigation Module for CineMatch V2.1.6

Provides keyboard navigation support, shortcuts, and focus
management for accessibility and power users.

Phase 6 - Task 6.8: Keyboard Navigation Support
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ModifierKey(Enum):
    """Keyboard modifier keys."""
    CTRL = "ctrl"
    ALT = "alt"
    SHIFT = "shift"
    META = "meta"  # Cmd on Mac, Win on Windows


class KeyCode(Enum):
    """Common key codes."""
    ENTER = "Enter"
    SPACE = "Space"
    ESCAPE = "Escape"
    TAB = "Tab"
    BACKSPACE = "Backspace"
    DELETE = "Delete"
    
    ARROW_UP = "ArrowUp"
    ARROW_DOWN = "ArrowDown"
    ARROW_LEFT = "ArrowLeft"
    ARROW_RIGHT = "ArrowRight"
    
    HOME = "Home"
    END = "End"
    PAGE_UP = "PageUp"
    PAGE_DOWN = "PageDown"
    
    F1 = "F1"
    F2 = "F2"
    F3 = "F3"
    F4 = "F4"
    F5 = "F5"
    F6 = "F6"
    F7 = "F7"
    F8 = "F8"
    F9 = "F9"
    F10 = "F10"
    F11 = "F11"
    F12 = "F12"


@dataclass
class KeyboardShortcut:
    """Definition of a keyboard shortcut."""
    key: str  # Key code or character
    modifiers: Set[ModifierKey] = field(default_factory=set)
    description: str = ""
    action_id: str = ""
    scope: str = "global"  # global, navigation, search, etc.
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "modifiers": [m.value for m in self.modifiers],
            "description": self.description,
            "action_id": self.action_id,
            "scope": self.scope,
            "enabled": self.enabled
        }
    
    def to_key_string(self) -> str:
        """Convert to display string like 'Ctrl+Shift+K'."""
        parts = []
        if ModifierKey.CTRL in self.modifiers:
            parts.append("Ctrl")
        if ModifierKey.ALT in self.modifiers:
            parts.append("Alt")
        if ModifierKey.SHIFT in self.modifiers:
            parts.append("Shift")
        if ModifierKey.META in self.modifiers:
            parts.append("⌘")  # Mac command key
        parts.append(self.key.upper() if len(self.key) == 1 else self.key)
        return "+".join(parts)
    
    def matches(
        self,
        key: str,
        ctrl: bool = False,
        alt: bool = False,
        shift: bool = False,
        meta: bool = False
    ) -> bool:
        """Check if this shortcut matches the given key combination."""
        if key.lower() != self.key.lower():
            return False
        
        expected_modifiers = {
            ModifierKey.CTRL: ctrl,
            ModifierKey.ALT: alt,
            ModifierKey.SHIFT: shift,
            ModifierKey.META: meta
        }
        
        for mod, pressed in expected_modifiers.items():
            if (mod in self.modifiers) != pressed:
                return False
        
        return True


@dataclass
class FocusTrap:
    """Configuration for a focus trap region."""
    container_id: str
    first_focusable_id: Optional[str] = None
    last_focusable_id: Optional[str] = None
    initial_focus_id: Optional[str] = None
    return_focus_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "container_id": self.container_id,
            "first_focusable_id": self.first_focusable_id,
            "last_focusable_id": self.last_focusable_id,
            "initial_focus_id": self.initial_focus_id,
            "return_focus_id": self.return_focus_id
        }


class DefaultShortcuts:
    """Default keyboard shortcuts for CineMatch."""
    
    # Navigation
    GO_HOME = KeyboardShortcut(
        key="h",
        modifiers={ModifierKey.ALT},
        description="Go to home page",
        action_id="nav.home",
        scope="navigation"
    )
    
    GO_SEARCH = KeyboardShortcut(
        key="s",
        modifiers={ModifierKey.ALT},
        description="Go to search",
        action_id="nav.search",
        scope="navigation"
    )
    
    GO_RECOMMENDATIONS = KeyboardShortcut(
        key="r",
        modifiers={ModifierKey.ALT},
        description="Go to recommendations",
        action_id="nav.recommendations",
        scope="navigation"
    )
    
    GO_PROFILE = KeyboardShortcut(
        key="p",
        modifiers={ModifierKey.ALT},
        description="Go to profile",
        action_id="nav.profile",
        scope="navigation"
    )
    
    GO_SETTINGS = KeyboardShortcut(
        key=",",
        modifiers={ModifierKey.CTRL},
        description="Open settings",
        action_id="nav.settings",
        scope="navigation"
    )
    
    # Search
    FOCUS_SEARCH = KeyboardShortcut(
        key="/",
        description="Focus search input",
        action_id="search.focus",
        scope="search"
    )
    
    CLEAR_SEARCH = KeyboardShortcut(
        key="Escape",
        description="Clear search",
        action_id="search.clear",
        scope="search"
    )
    
    # Actions
    REFRESH = KeyboardShortcut(
        key="r",
        modifiers={ModifierKey.CTRL},
        description="Refresh content",
        action_id="action.refresh",
        scope="global"
    )
    
    CLOSE = KeyboardShortcut(
        key="Escape",
        description="Close dialog/modal",
        action_id="action.close",
        scope="global"
    )
    
    SUBMIT = KeyboardShortcut(
        key="Enter",
        modifiers={ModifierKey.CTRL},
        description="Submit form",
        action_id="action.submit",
        scope="global"
    )
    
    # Accessibility
    SHOW_SHORTCUTS = KeyboardShortcut(
        key="?",
        modifiers={ModifierKey.SHIFT},
        description="Show keyboard shortcuts",
        action_id="accessibility.shortcuts",
        scope="global"
    )
    
    SKIP_TO_MAIN = KeyboardShortcut(
        key="m",
        modifiers={ModifierKey.ALT},
        description="Skip to main content",
        action_id="accessibility.skip_main",
        scope="global"
    )
    
    # List navigation
    NEXT_ITEM = KeyboardShortcut(
        key="ArrowDown",
        description="Next item",
        action_id="list.next",
        scope="list"
    )
    
    PREV_ITEM = KeyboardShortcut(
        key="ArrowUp",
        description="Previous item",
        action_id="list.prev",
        scope="list"
    )
    
    SELECT_ITEM = KeyboardShortcut(
        key="Enter",
        description="Select item",
        action_id="list.select",
        scope="list"
    )
    
    FIRST_ITEM = KeyboardShortcut(
        key="Home",
        description="Go to first item",
        action_id="list.first",
        scope="list"
    )
    
    LAST_ITEM = KeyboardShortcut(
        key="End",
        description="Go to last item",
        action_id="list.last",
        scope="list"
    )
    
    @classmethod
    def get_all(cls) -> List[KeyboardShortcut]:
        """Get all default shortcuts."""
        return [
            cls.GO_HOME, cls.GO_SEARCH, cls.GO_RECOMMENDATIONS,
            cls.GO_PROFILE, cls.GO_SETTINGS,
            cls.FOCUS_SEARCH, cls.CLEAR_SEARCH,
            cls.REFRESH, cls.CLOSE, cls.SUBMIT,
            cls.SHOW_SHORTCUTS, cls.SKIP_TO_MAIN,
            cls.NEXT_ITEM, cls.PREV_ITEM, cls.SELECT_ITEM,
            cls.FIRST_ITEM, cls.LAST_ITEM
        ]
    
    @classmethod
    def get_by_scope(cls, scope: str) -> List[KeyboardShortcut]:
        """Get shortcuts for a specific scope."""
        return [s for s in cls.get_all() if s.scope == scope]


class KeyboardShortcutManager:
    """Manages keyboard shortcuts."""
    
    def __init__(self):
        self._shortcuts: Dict[str, KeyboardShortcut] = {}
        self._handlers: Dict[str, Callable[[], None]] = {}
        self._enabled = True
        
        # Register default shortcuts
        for shortcut in DefaultShortcuts.get_all():
            self.register(shortcut)
    
    def register(self, shortcut: KeyboardShortcut) -> None:
        """Register a keyboard shortcut."""
        self._shortcuts[shortcut.action_id] = shortcut
    
    def unregister(self, action_id: str) -> bool:
        """Unregister a keyboard shortcut."""
        if action_id in self._shortcuts:
            del self._shortcuts[action_id]
            return True
        return False
    
    def set_handler(
        self,
        action_id: str,
        handler: Callable[[], None]
    ) -> None:
        """Set handler for a shortcut action."""
        self._handlers[action_id] = handler
    
    def get_shortcut(self, action_id: str) -> Optional[KeyboardShortcut]:
        """Get shortcut by action ID."""
        return self._shortcuts.get(action_id)
    
    def list_shortcuts(
        self,
        scope: Optional[str] = None,
        enabled_only: bool = True
    ) -> List[KeyboardShortcut]:
        """List registered shortcuts."""
        shortcuts = list(self._shortcuts.values())
        
        if scope:
            shortcuts = [s for s in shortcuts if s.scope == scope]
        
        if enabled_only:
            shortcuts = [s for s in shortcuts if s.enabled]
        
        return shortcuts
    
    def find_shortcut(
        self,
        key: str,
        ctrl: bool = False,
        alt: bool = False,
        shift: bool = False,
        meta: bool = False
    ) -> Optional[KeyboardShortcut]:
        """Find shortcut matching key combination."""
        for shortcut in self._shortcuts.values():
            if shortcut.enabled and shortcut.matches(key, ctrl, alt, shift, meta):
                return shortcut
        return None
    
    def handle_key(
        self,
        key: str,
        ctrl: bool = False,
        alt: bool = False,
        shift: bool = False,
        meta: bool = False
    ) -> bool:
        """Handle a key press and execute action if matched."""
        if not self._enabled:
            return False
        
        shortcut = self.find_shortcut(key, ctrl, alt, shift, meta)
        if shortcut and shortcut.action_id in self._handlers:
            try:
                self._handlers[shortcut.action_id]()
                return True
            except Exception as e:
                logger.error(f"Shortcut handler error: {e}")
        
        return False
    
    def enable(self) -> None:
        """Enable keyboard shortcuts."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable keyboard shortcuts."""
        self._enabled = False
    
    def update_shortcut(
        self,
        action_id: str,
        key: Optional[str] = None,
        modifiers: Optional[Set[ModifierKey]] = None
    ) -> bool:
        """Update shortcut key binding."""
        shortcut = self._shortcuts.get(action_id)
        if shortcut:
            if key is not None:
                shortcut.key = key
            if modifiers is not None:
                shortcut.modifiers = modifiers
            return True
        return False
    
    def export_shortcuts(self) -> Dict[str, Any]:
        """Export shortcuts as JSON-serializable dict."""
        return {
            action_id: shortcut.to_dict()
            for action_id, shortcut in self._shortcuts.items()
        }
    
    def generate_help_html(self) -> str:
        """Generate HTML help for keyboard shortcuts."""
        scopes = {}
        for shortcut in self._shortcuts.values():
            if shortcut.enabled:
                if shortcut.scope not in scopes:
                    scopes[shortcut.scope] = []
                scopes[shortcut.scope].append(shortcut)
        
        html = '<div class="keyboard-shortcuts-help" role="dialog" aria-label="Keyboard shortcuts">'
        html += '<h2>Keyboard Shortcuts</h2>'
        
        scope_names = {
            "global": "Global",
            "navigation": "Navigation",
            "search": "Search",
            "list": "List Navigation"
        }
        
        for scope, shortcuts in scopes.items():
            html += f'<section><h3>{scope_names.get(scope, scope.title())}</h3>'
            html += '<table role="grid"><tbody>'
            
            for shortcut in shortcuts:
                html += f'''
                <tr role="row">
                    <td role="gridcell"><kbd>{shortcut.to_key_string()}</kbd></td>
                    <td role="gridcell">{shortcut.description}</td>
                </tr>
                '''
            
            html += '</tbody></table></section>'
        
        html += '</div>'
        return html


class FocusManager:
    """Manages focus for keyboard navigation."""
    
    def __init__(self):
        self._focus_traps: Dict[str, FocusTrap] = {}
        self._active_trap: Optional[str] = None
        self._focus_history: List[str] = []
    
    def register_trap(self, trap: FocusTrap) -> None:
        """Register a focus trap region."""
        self._focus_traps[trap.container_id] = trap
    
    def unregister_trap(self, container_id: str) -> bool:
        """Unregister a focus trap."""
        if container_id in self._focus_traps:
            del self._focus_traps[container_id]
            if self._active_trap == container_id:
                self._active_trap = None
            return True
        return False
    
    def activate_trap(self, container_id: str) -> bool:
        """Activate a focus trap."""
        if container_id in self._focus_traps:
            self._active_trap = container_id
            return True
        return False
    
    def deactivate_trap(self) -> Optional[str]:
        """Deactivate current focus trap."""
        trap_id = self._active_trap
        self._active_trap = None
        return trap_id
    
    def get_active_trap(self) -> Optional[FocusTrap]:
        """Get active focus trap."""
        if self._active_trap:
            return self._focus_traps.get(self._active_trap)
        return None
    
    def push_focus(self, element_id: str) -> None:
        """Push element to focus history."""
        self._focus_history.append(element_id)
        if len(self._focus_history) > 50:
            self._focus_history = self._focus_history[-50:]
    
    def pop_focus(self) -> Optional[str]:
        """Pop and return last focused element."""
        if self._focus_history:
            return self._focus_history.pop()
        return None
    
    def clear_history(self) -> None:
        """Clear focus history."""
        self._focus_history.clear()
    
    def generate_focus_trap_js(self, trap: FocusTrap) -> str:
        """Generate JavaScript for focus trap."""
        return f'''
        <script>
        (function() {{
            const container = document.getElementById('{trap.container_id}');
            if (!container) return;
            
            const focusableElements = container.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            
            if (focusableElements.length === 0) return;
            
            const firstFocusable = focusableElements[0];
            const lastFocusable = focusableElements[focusableElements.length - 1];
            
            container.addEventListener('keydown', function(e) {{
                if (e.key !== 'Tab') return;
                
                if (e.shiftKey) {{
                    if (document.activeElement === firstFocusable) {{
                        e.preventDefault();
                        lastFocusable.focus();
                    }}
                }} else {{
                    if (document.activeElement === lastFocusable) {{
                        e.preventDefault();
                        firstFocusable.focus();
                    }}
                }}
            }});
            
            // Set initial focus
            {'document.getElementById("' + trap.initial_focus_id + '").focus();' if trap.initial_focus_id else 'firstFocusable.focus();'}
        }})();
        </script>
        '''


class RovingTabIndex:
    """Implements roving tabindex for composite widgets."""
    
    @staticmethod
    def generate_js(
        container_id: str,
        item_selector: str = '[role="option"], [role="menuitem"], [role="tab"]',
        orientation: str = "vertical"  # vertical, horizontal, both
    ) -> str:
        """Generate JavaScript for roving tabindex."""
        key_map = {
            "vertical": "{'ArrowUp': -1, 'ArrowDown': 1}",
            "horizontal": "{'ArrowLeft': -1, 'ArrowRight': 1}",
            "both": "{'ArrowUp': -1, 'ArrowDown': 1, 'ArrowLeft': -1, 'ArrowRight': 1}"
        }
        
        return f'''
        <script>
        (function() {{
            const container = document.getElementById('{container_id}');
            if (!container) return;
            
            const items = container.querySelectorAll('{item_selector}');
            if (items.length === 0) return;
            
            const keyMap = {key_map.get(orientation, key_map["vertical"])};
            let currentIndex = 0;
            
            // Initialize tabindex
            items.forEach((item, index) => {{
                item.setAttribute('tabindex', index === 0 ? '0' : '-1');
            }});
            
            function focusItem(index) {{
                items[currentIndex].setAttribute('tabindex', '-1');
                currentIndex = index;
                items[currentIndex].setAttribute('tabindex', '0');
                items[currentIndex].focus();
            }}
            
            container.addEventListener('keydown', function(e) {{
                const direction = keyMap[e.key];
                if (direction === undefined) return;
                
                e.preventDefault();
                let newIndex = currentIndex + direction;
                
                // Wrap around
                if (newIndex < 0) newIndex = items.length - 1;
                if (newIndex >= items.length) newIndex = 0;
                
                focusItem(newIndex);
            }});
            
            // Handle Home/End
            container.addEventListener('keydown', function(e) {{
                if (e.key === 'Home') {{
                    e.preventDefault();
                    focusItem(0);
                }} else if (e.key === 'End') {{
                    e.preventDefault();
                    focusItem(items.length - 1);
                }}
            }});
        }})();
        </script>
        '''


class KeyboardNavigationHelper:
    """Helper utilities for keyboard navigation."""
    
    @staticmethod
    def generate_keyboard_handler_js() -> str:
        """Generate global keyboard handler JavaScript."""
        return '''
        <script>
        window.CineMatchKeyboard = {
            handlers: {},
            
            register: function(key, modifiers, handler) {
                const id = this._makeId(key, modifiers);
                this.handlers[id] = handler;
            },
            
            unregister: function(key, modifiers) {
                const id = this._makeId(key, modifiers);
                delete this.handlers[id];
            },
            
            _makeId: function(key, modifiers) {
                const mods = [];
                if (modifiers.ctrl) mods.push('ctrl');
                if (modifiers.alt) mods.push('alt');
                if (modifiers.shift) mods.push('shift');
                if (modifiers.meta) mods.push('meta');
                mods.push(key.toLowerCase());
                return mods.join('+');
            },
            
            handleKeyDown: function(e) {
                // Skip if in input field
                if (['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName)) {
                    if (!e.ctrlKey && !e.altKey && !e.metaKey) return;
                }
                
                const id = this._makeId(e.key, {
                    ctrl: e.ctrlKey,
                    alt: e.altKey,
                    shift: e.shiftKey,
                    meta: e.metaKey
                });
                
                const handler = this.handlers[id];
                if (handler) {
                    e.preventDefault();
                    handler(e);
                }
            }
        };
        
        document.addEventListener('keydown', function(e) {
            window.CineMatchKeyboard.handleKeyDown(e);
        });
        </script>
        '''
    
    @staticmethod
    def generate_skip_links(links: List[Tuple[str, str]]) -> str:
        """Generate skip links HTML."""
        html = '<nav class="skip-links" aria-label="Skip links">'
        
        for href, text in links:
            html += f'''
            <a href="#{href}" class="skip-link">
                {text}
            </a>
            '''
        
        html += '</nav>'
        html += '''
        <style>
            .skip-links {
                position: absolute;
                top: 0;
                left: 0;
                z-index: 9999;
            }
            .skip-link {
                position: absolute;
                left: -9999px;
                padding: 1em;
                background: #000;
                color: #fff;
                text-decoration: underline;
            }
            .skip-link:focus {
                left: 0;
            }
        </style>
        '''
        
        return html
    
    @staticmethod
    def generate_focus_indicator_css() -> str:
        """Generate CSS for focus indicators."""
        return '''
        <style>
            /* Focus ring styles */
            :focus {
                outline: 2px solid #005fcc;
                outline-offset: 2px;
            }
            
            :focus:not(:focus-visible) {
                outline: none;
            }
            
            :focus-visible {
                outline: 2px solid #005fcc;
                outline-offset: 2px;
            }
            
            /* Enhanced focus for interactive elements */
            button:focus-visible,
            [role="button"]:focus-visible,
            a:focus-visible {
                outline: 3px solid #005fcc;
                box-shadow: 0 0 0 6px rgba(0, 95, 204, 0.2);
            }
            
            /* Focus within for groups */
            [role="listbox"]:focus-within,
            [role="menu"]:focus-within,
            [role="tablist"]:focus-within {
                outline: 1px solid #005fcc;
            }
        </style>
        '''


# Global instances
_shortcut_manager: Optional[KeyboardShortcutManager] = None
_focus_manager: Optional[FocusManager] = None


def get_shortcut_manager() -> KeyboardShortcutManager:
    """Get global shortcut manager."""
    global _shortcut_manager
    if _shortcut_manager is None:
        _shortcut_manager = KeyboardShortcutManager()
    return _shortcut_manager


def get_focus_manager() -> FocusManager:
    """Get global focus manager."""
    global _focus_manager
    if _focus_manager is None:
        _focus_manager = FocusManager()
    return _focus_manager


def register_shortcut(shortcut: KeyboardShortcut) -> None:
    """Register a keyboard shortcut."""
    get_shortcut_manager().register(shortcut)


def on_shortcut(action_id: str, handler: Callable[[], None]) -> None:
    """Register handler for shortcut action."""
    get_shortcut_manager().set_handler(action_id, handler)
