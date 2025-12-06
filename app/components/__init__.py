"""
CineMatch V2.1.6 - UI Components Module

Reusable Streamlit UI components for consistent user experience.
Includes loading spinners, error messages, help tooltips, and accessibility features.

Author: CineMatch Development Team
Date: December 5, 2025
"""

from .ui_helpers import (
    # Loading spinners (Task 2.1)
    show_loading_spinner,
    show_progress_bar,
    show_model_loading_status,
    
    # Error messages (Task 2.2)
    show_friendly_error,
    show_validation_error,
    show_system_error,
    
    # Input validation UI (Task 2.3)
    validated_number_input,
    validated_text_input,
    
    # Help tooltips (Task 2.4)
    show_algorithm_help,
    show_feature_tooltip,
    create_info_callout,
    
    # Mobile responsiveness (Task 2.5)
    inject_mobile_responsive_css,
    
    # Accessibility (Task 2.8)
    accessible_button,
    accessible_metric,
    screen_reader_text,
    
    # Export functionality (Task 2.12)
    export_to_csv,
    export_to_json,
    create_download_button,
    
    # Preference memory (Task 2.11)
    save_user_preference,
    load_user_preference,
    get_default_algorithm,
    
    # Visualization helpers (Task 2.13)
    create_chart_title,
    format_axis_label,
    
    # Theme support (Task 2.14)
    inject_theme_css,
    toggle_dark_mode,
)

from .onboarding import (
    # Onboarding flow (Task 2.7)
    show_onboarding_modal,
    show_feature_tour,
    show_welcome_banner,
    check_first_visit,
    reset_onboarding,
    
    # Keyboard shortcuts (Task 2.6)
    show_keyboard_shortcuts,
)

__all__ = [
    # Loading
    'show_loading_spinner',
    'show_progress_bar', 
    'show_model_loading_status',
    
    # Errors
    'show_friendly_error',
    'show_validation_error',
    'show_system_error',
    
    # Input validation
    'validated_number_input',
    'validated_text_input',
    
    # Help
    'show_algorithm_help',
    'show_feature_tooltip',
    'create_info_callout',
    
    # Mobile
    'inject_mobile_responsive_css',
    
    # Accessibility
    'accessible_button',
    'accessible_metric',
    'screen_reader_text',
    
    # Export
    'export_to_csv',
    'export_to_json',
    'create_download_button',
    
    # Preferences
    'save_user_preference',
    'load_user_preference',
    'get_default_algorithm',
    
    # Visualization
    'create_chart_title',
    'format_axis_label',
    
    # Theme
    'inject_theme_css',
    'toggle_dark_mode',
    
    # Onboarding
    'show_onboarding_modal',
    'show_feature_tour',
    'show_welcome_banner',
    'check_first_visit',
    'reset_onboarding',
    'show_keyboard_shortcuts',
]
