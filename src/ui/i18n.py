"""
Internationalization (i18n) Module for CineMatch V2.1.6

Provides multi-language support, translations, and locale
management for the application.

Phase 6 - Task 6.6: i18n Support
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class LocaleInfo:
    """Information about a locale."""
    code: str  # e.g., 'en-US', 'es-ES', 'fr-FR'
    language: str  # e.g., 'en', 'es', 'fr'
    region: Optional[str] = None  # e.g., 'US', 'ES', 'FR'
    name: str = ""  # e.g., 'English (US)'
    native_name: str = ""  # e.g., 'English'
    direction: str = "ltr"  # 'ltr' or 'rtl'
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    number_decimal: str = "."
    number_thousands: str = ","
    currency_symbol: str = "$"
    currency_code: str = "USD"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "language": self.language,
            "region": self.region,
            "name": self.name,
            "native_name": self.native_name,
            "direction": self.direction,
            "date_format": self.date_format,
            "time_format": self.time_format,
            "datetime_format": self.datetime_format,
            "number_decimal": self.number_decimal,
            "number_thousands": self.number_thousands,
            "currency_symbol": self.currency_symbol,
            "currency_code": self.currency_code
        }


class BuiltinLocales:
    """Built-in locale definitions."""
    
    EN_US = LocaleInfo(
        code="en-US",
        language="en",
        region="US",
        name="English (US)",
        native_name="English",
        date_format="%m/%d/%Y",
        number_decimal=".",
        number_thousands=",",
        currency_symbol="$",
        currency_code="USD"
    )
    
    EN_GB = LocaleInfo(
        code="en-GB",
        language="en",
        region="GB",
        name="English (UK)",
        native_name="English",
        date_format="%d/%m/%Y",
        number_decimal=".",
        number_thousands=",",
        currency_symbol="£",
        currency_code="GBP"
    )
    
    ES_ES = LocaleInfo(
        code="es-ES",
        language="es",
        region="ES",
        name="Spanish (Spain)",
        native_name="Español",
        date_format="%d/%m/%Y",
        number_decimal=",",
        number_thousands=".",
        currency_symbol="€",
        currency_code="EUR"
    )
    
    FR_FR = LocaleInfo(
        code="fr-FR",
        language="fr",
        region="FR",
        name="French (France)",
        native_name="Français",
        date_format="%d/%m/%Y",
        number_decimal=",",
        number_thousands=" ",
        currency_symbol="€",
        currency_code="EUR"
    )
    
    DE_DE = LocaleInfo(
        code="de-DE",
        language="de",
        region="DE",
        name="German (Germany)",
        native_name="Deutsch",
        date_format="%d.%m.%Y",
        number_decimal=",",
        number_thousands=".",
        currency_symbol="€",
        currency_code="EUR"
    )
    
    JA_JP = LocaleInfo(
        code="ja-JP",
        language="ja",
        region="JP",
        name="Japanese",
        native_name="日本語",
        date_format="%Y/%m/%d",
        number_decimal=".",
        number_thousands=",",
        currency_symbol="¥",
        currency_code="JPY"
    )
    
    ZH_CN = LocaleInfo(
        code="zh-CN",
        language="zh",
        region="CN",
        name="Chinese (Simplified)",
        native_name="简体中文",
        date_format="%Y/%m/%d",
        number_decimal=".",
        number_thousands=",",
        currency_symbol="¥",
        currency_code="CNY"
    )
    
    AR_SA = LocaleInfo(
        code="ar-SA",
        language="ar",
        region="SA",
        name="Arabic (Saudi Arabia)",
        native_name="العربية",
        direction="rtl",
        date_format="%d/%m/%Y",
        number_decimal="٫",
        number_thousands="٬",
        currency_symbol="ر.س",
        currency_code="SAR"
    )
    
    PT_BR = LocaleInfo(
        code="pt-BR",
        language="pt",
        region="BR",
        name="Portuguese (Brazil)",
        native_name="Português",
        date_format="%d/%m/%Y",
        number_decimal=",",
        number_thousands=".",
        currency_symbol="R$",
        currency_code="BRL"
    )
    
    @classmethod
    def get_all(cls) -> List[LocaleInfo]:
        """Get all built-in locales."""
        return [
            cls.EN_US, cls.EN_GB, cls.ES_ES, cls.FR_FR,
            cls.DE_DE, cls.JA_JP, cls.ZH_CN, cls.AR_SA, cls.PT_BR
        ]
    
    @classmethod
    def get_by_code(cls, code: str) -> Optional[LocaleInfo]:
        """Get locale by code."""
        for locale in cls.get_all():
            if locale.code == code:
                return locale
        return None


@dataclass
class TranslationEntry:
    """A translation entry with metadata."""
    key: str
    value: str
    context: Optional[str] = None
    plural_forms: Optional[Dict[str, str]] = None


class TranslationCatalog:
    """Manages translations for a single locale."""
    
    def __init__(self, locale_code: str):
        self.locale_code = locale_code
        self._translations: Dict[str, TranslationEntry] = {}
        self._fallback: Optional['TranslationCatalog'] = None
    
    def add(
        self,
        key: str,
        value: str,
        context: Optional[str] = None,
        plural_forms: Optional[Dict[str, str]] = None
    ) -> None:
        """Add a translation."""
        self._translations[key] = TranslationEntry(
            key=key,
            value=value,
            context=context,
            plural_forms=plural_forms
        )
    
    def add_all(self, translations: Dict[str, str]) -> None:
        """Add multiple translations."""
        for key, value in translations.items():
            self.add(key, value)
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a translation."""
        entry = self._translations.get(key)
        if entry:
            return entry.value
        if self._fallback:
            return self._fallback.get(key, default)
        return default
    
    def get_plural(
        self,
        key: str,
        count: int,
        default: Optional[str] = None
    ) -> Optional[str]:
        """Get a pluralized translation."""
        entry = self._translations.get(key)
        if entry and entry.plural_forms:
            # Simple plural rules
            if count == 0 and "zero" in entry.plural_forms:
                return entry.plural_forms["zero"]
            elif count == 1 and "one" in entry.plural_forms:
                return entry.plural_forms["one"]
            elif "many" in entry.plural_forms:
                return entry.plural_forms["many"]
        if entry:
            return entry.value
        if self._fallback:
            return self._fallback.get_plural(key, count, default)
        return default
    
    def set_fallback(self, fallback: 'TranslationCatalog') -> None:
        """Set fallback catalog for missing translations."""
        self._fallback = fallback
    
    def has_key(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._translations
    
    def keys(self) -> List[str]:
        """Get all translation keys."""
        return list(self._translations.keys())
    
    def to_dict(self) -> Dict[str, str]:
        """Export translations as dictionary."""
        return {k: v.value for k, v in self._translations.items()}
    
    @classmethod
    def from_dict(cls, locale_code: str, data: Dict[str, str]) -> 'TranslationCatalog':
        """Create catalog from dictionary."""
        catalog = cls(locale_code)
        catalog.add_all(data)
        return catalog
    
    @classmethod
    def from_json_file(cls, locale_code: str, path: Path) -> 'TranslationCatalog':
        """Load catalog from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(locale_code, data)


# Built-in English translations
ENGLISH_TRANSLATIONS = {
    # General
    "app.title": "CineMatch",
    "app.subtitle": "AI-Powered Movie Recommendations",
    "app.loading": "Loading...",
    "app.error": "An error occurred",
    "app.success": "Success!",
    
    # Navigation
    "nav.home": "Home",
    "nav.recommendations": "Recommendations",
    "nav.search": "Search",
    "nav.profile": "Profile",
    "nav.settings": "Settings",
    "nav.about": "About",
    
    # Recommendations
    "rec.title": "Your Recommendations",
    "rec.no_results": "No recommendations found",
    "rec.loading": "Generating recommendations...",
    "rec.algorithm": "Algorithm",
    "rec.count": "Number of recommendations",
    "rec.refresh": "Refresh",
    "rec.why": "Why this recommendation?",
    
    # Search
    "search.placeholder": "Search for movies...",
    "search.results": "Search Results",
    "search.no_results": "No movies found",
    "search.filters": "Filters",
    "search.genre": "Genre",
    "search.year": "Year",
    "search.rating": "Rating",
    
    # Movies
    "movie.details": "Movie Details",
    "movie.rating": "Rating",
    "movie.genres": "Genres",
    "movie.year": "Year",
    "movie.similar": "Similar Movies",
    "movie.rate": "Rate this movie",
    "movie.add_watchlist": "Add to Watchlist",
    "movie.remove_watchlist": "Remove from Watchlist",
    
    # User
    "user.profile": "User Profile",
    "user.ratings": "Your Ratings",
    "user.watchlist": "Watchlist",
    "user.history": "History",
    "user.preferences": "Preferences",
    
    # Settings
    "settings.title": "Settings",
    "settings.language": "Language",
    "settings.theme": "Theme",
    "settings.notifications": "Notifications",
    "settings.privacy": "Privacy",
    "settings.save": "Save Changes",
    "settings.reset": "Reset to Defaults",
    
    # Errors
    "error.generic": "Something went wrong. Please try again.",
    "error.not_found": "Not found",
    "error.network": "Network error. Please check your connection.",
    "error.unauthorized": "Please log in to continue.",
    
    # Actions
    "action.submit": "Submit",
    "action.cancel": "Cancel",
    "action.save": "Save",
    "action.delete": "Delete",
    "action.edit": "Edit",
    "action.close": "Close",
    "action.confirm": "Confirm",
    "action.back": "Back",
    "action.next": "Next",
    
    # Messages
    "msg.saved": "Changes saved successfully",
    "msg.deleted": "Item deleted",
    "msg.confirm_delete": "Are you sure you want to delete this?",
    
    # Analytics
    "analytics.title": "Analytics",
    "analytics.overview": "Overview",
    "analytics.recommendations_count": "Recommendations Generated",
    "analytics.active_users": "Active Users",
    "analytics.popular_genres": "Popular Genres",
    "analytics.top_movies": "Top Rated Movies"
}

# Spanish translations
SPANISH_TRANSLATIONS = {
    "app.title": "CineMatch",
    "app.subtitle": "Recomendaciones de Películas con IA",
    "app.loading": "Cargando...",
    "app.error": "Ocurrió un error",
    "app.success": "¡Éxito!",
    
    "nav.home": "Inicio",
    "nav.recommendations": "Recomendaciones",
    "nav.search": "Buscar",
    "nav.profile": "Perfil",
    "nav.settings": "Configuración",
    "nav.about": "Acerca de",
    
    "rec.title": "Tus Recomendaciones",
    "rec.no_results": "No se encontraron recomendaciones",
    "rec.loading": "Generando recomendaciones...",
    "rec.algorithm": "Algoritmo",
    "rec.count": "Número de recomendaciones",
    "rec.refresh": "Actualizar",
    "rec.why": "¿Por qué esta recomendación?",
    
    "search.placeholder": "Buscar películas...",
    "search.results": "Resultados de Búsqueda",
    "search.no_results": "No se encontraron películas",
    "search.filters": "Filtros",
    "search.genre": "Género",
    "search.year": "Año",
    "search.rating": "Calificación",
    
    "movie.details": "Detalles de la Película",
    "movie.rating": "Calificación",
    "movie.genres": "Géneros",
    "movie.year": "Año",
    "movie.similar": "Películas Similares",
    "movie.rate": "Califica esta película",
    
    "settings.title": "Configuración",
    "settings.language": "Idioma",
    "settings.theme": "Tema",
    "settings.save": "Guardar Cambios",
    
    "action.submit": "Enviar",
    "action.cancel": "Cancelar",
    "action.save": "Guardar",
    "action.delete": "Eliminar",
    "action.close": "Cerrar",
    "action.back": "Atrás",
    "action.next": "Siguiente"
}

# French translations
FRENCH_TRANSLATIONS = {
    "app.title": "CineMatch",
    "app.subtitle": "Recommandations de Films par IA",
    "app.loading": "Chargement...",
    "app.error": "Une erreur s'est produite",
    "app.success": "Succès !",
    
    "nav.home": "Accueil",
    "nav.recommendations": "Recommandations",
    "nav.search": "Rechercher",
    "nav.profile": "Profil",
    "nav.settings": "Paramètres",
    "nav.about": "À propos",
    
    "rec.title": "Vos Recommandations",
    "rec.no_results": "Aucune recommandation trouvée",
    "rec.loading": "Génération des recommandations...",
    "rec.algorithm": "Algorithme",
    "rec.count": "Nombre de recommandations",
    "rec.refresh": "Actualiser",
    
    "search.placeholder": "Rechercher des films...",
    "search.results": "Résultats de Recherche",
    "search.no_results": "Aucun film trouvé",
    "search.filters": "Filtres",
    "search.genre": "Genre",
    "search.year": "Année",
    "search.rating": "Note",
    
    "movie.details": "Détails du Film",
    "movie.rating": "Note",
    "movie.genres": "Genres",
    "movie.year": "Année",
    "movie.similar": "Films Similaires",
    
    "settings.title": "Paramètres",
    "settings.language": "Langue",
    "settings.theme": "Thème",
    "settings.save": "Enregistrer",
    
    "action.submit": "Soumettre",
    "action.cancel": "Annuler",
    "action.save": "Enregistrer",
    "action.delete": "Supprimer",
    "action.close": "Fermer"
}


class I18nManager:
    """Manages internationalization."""
    
    def __init__(self, translations_dir: Optional[Path] = None):
        self._locales: Dict[str, LocaleInfo] = {}
        self._catalogs: Dict[str, TranslationCatalog] = {}
        self._current_locale: LocaleInfo = BuiltinLocales.EN_US
        self._default_locale: LocaleInfo = BuiltinLocales.EN_US
        self._translations_dir = translations_dir
        self._change_callbacks: List[Callable[[LocaleInfo], None]] = []
        
        # Load built-in locales
        for locale in BuiltinLocales.get_all():
            self._locales[locale.code] = locale
        
        # Load built-in translations
        self._load_builtin_translations()
        
        # Load translations from directory
        if translations_dir and translations_dir.exists():
            self._load_translations_from_directory(translations_dir)
    
    def _load_builtin_translations(self) -> None:
        """Load built-in translations."""
        en_catalog = TranslationCatalog.from_dict("en-US", ENGLISH_TRANSLATIONS)
        self._catalogs["en-US"] = en_catalog
        self._catalogs["en-GB"] = en_catalog  # Share English translations
        
        es_catalog = TranslationCatalog.from_dict("es-ES", SPANISH_TRANSLATIONS)
        es_catalog.set_fallback(en_catalog)
        self._catalogs["es-ES"] = es_catalog
        
        fr_catalog = TranslationCatalog.from_dict("fr-FR", FRENCH_TRANSLATIONS)
        fr_catalog.set_fallback(en_catalog)
        self._catalogs["fr-FR"] = fr_catalog
    
    def _load_translations_from_directory(self, directory: Path) -> None:
        """Load translations from JSON files."""
        for path in directory.glob("*.json"):
            locale_code = path.stem
            try:
                catalog = TranslationCatalog.from_json_file(locale_code, path)
                # Set English as fallback
                if "en-US" in self._catalogs:
                    catalog.set_fallback(self._catalogs["en-US"])
                self._catalogs[locale_code] = catalog
                logger.info(f"Loaded translations for {locale_code}")
            except Exception as e:
                logger.error(f"Failed to load translations from {path}: {e}")
    
    def get_locale(self, code: str) -> Optional[LocaleInfo]:
        """Get locale by code."""
        return self._locales.get(code)
    
    def list_locales(self) -> List[LocaleInfo]:
        """List all available locales."""
        return list(self._locales.values())
    
    def set_locale(self, code: str) -> bool:
        """Set current locale."""
        locale = self._locales.get(code)
        if locale:
            self._current_locale = locale
            for callback in self._change_callbacks:
                try:
                    callback(locale)
                except Exception as e:
                    logger.error(f"Locale change callback error: {e}")
            return True
        return False
    
    def get_current_locale(self) -> LocaleInfo:
        """Get current locale."""
        return self._current_locale
    
    def on_locale_change(self, callback: Callable[[LocaleInfo], None]) -> None:
        """Register callback for locale changes."""
        self._change_callbacks.append(callback)
    
    def translate(
        self,
        key: str,
        default: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Get translation for key.
        
        Args:
            key: Translation key
            default: Default value if key not found
            **kwargs: Variables for interpolation
            
        Returns:
            Translated string
        """
        catalog = self._catalogs.get(self._current_locale.code)
        if not catalog:
            # Try language-only fallback
            catalog = self._catalogs.get(f"{self._current_locale.language}-US")
        
        if catalog:
            value = catalog.get(key)
            if value:
                # Interpolate variables
                if kwargs:
                    try:
                        value = value.format(**kwargs)
                    except KeyError:
                        pass
                return value
        
        return default or key
    
    def translate_plural(
        self,
        key: str,
        count: int,
        default: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Get pluralized translation.
        
        Args:
            key: Translation key
            count: Count for pluralization
            default: Default value if key not found
            **kwargs: Variables for interpolation
        """
        catalog = self._catalogs.get(self._current_locale.code)
        if catalog:
            value = catalog.get_plural(key, count, default)
            if value:
                kwargs["count"] = count
                try:
                    value = value.format(**kwargs)
                except KeyError:
                    pass
                return value
        return default or key
    
    def format_date(self, date: datetime) -> str:
        """Format date according to current locale."""
        return date.strftime(self._current_locale.date_format)
    
    def format_time(self, time: datetime) -> str:
        """Format time according to current locale."""
        return time.strftime(self._current_locale.time_format)
    
    def format_datetime(self, dt: datetime) -> str:
        """Format datetime according to current locale."""
        return dt.strftime(self._current_locale.datetime_format)
    
    def format_number(
        self,
        number: Union[int, float, Decimal],
        decimals: int = 2
    ) -> str:
        """Format number according to current locale."""
        locale = self._current_locale
        
        if isinstance(number, int):
            formatted = str(number)
        else:
            formatted = f"{number:.{decimals}f}"
        
        # Split integer and decimal parts
        parts = formatted.split(".")
        integer_part = parts[0]
        decimal_part = parts[1] if len(parts) > 1 else ""
        
        # Add thousands separator
        if len(integer_part) > 3:
            groups = []
            while integer_part:
                groups.append(integer_part[-3:])
                integer_part = integer_part[:-3]
            integer_part = locale.number_thousands.join(reversed(groups))
        
        if decimal_part:
            return f"{integer_part}{locale.number_decimal}{decimal_part}"
        return integer_part
    
    def format_currency(
        self,
        amount: Union[int, float, Decimal],
        currency_code: Optional[str] = None
    ) -> str:
        """Format currency according to current locale."""
        locale = self._current_locale
        symbol = locale.currency_symbol
        formatted = self.format_number(amount, 2)
        return f"{symbol}{formatted}"
    
    def get_catalog(self, locale_code: Optional[str] = None) -> Optional[TranslationCatalog]:
        """Get translation catalog for locale."""
        code = locale_code or self._current_locale.code
        return self._catalogs.get(code)
    
    def add_translations(
        self,
        locale_code: str,
        translations: Dict[str, str]
    ) -> None:
        """Add translations to a locale."""
        if locale_code not in self._catalogs:
            self._catalogs[locale_code] = TranslationCatalog(locale_code)
            # Set fallback
            if "en-US" in self._catalogs:
                self._catalogs[locale_code].set_fallback(self._catalogs["en-US"])
        
        self._catalogs[locale_code].add_all(translations)


# Translation function shortcuts
_i18n_manager: Optional[I18nManager] = None


def get_i18n_manager(translations_dir: Optional[Path] = None) -> I18nManager:
    """Get global i18n manager instance."""
    global _i18n_manager
    if _i18n_manager is None:
        _i18n_manager = I18nManager(translations_dir)
    return _i18n_manager


def t(key: str, default: Optional[str] = None, **kwargs: Any) -> str:
    """Shortcut for translation."""
    return get_i18n_manager().translate(key, default, **kwargs)


def tn(key: str, count: int, default: Optional[str] = None, **kwargs: Any) -> str:
    """Shortcut for plural translation."""
    return get_i18n_manager().translate_plural(key, count, default, **kwargs)


def set_locale(code: str) -> bool:
    """Shortcut for setting locale."""
    return get_i18n_manager().set_locale(code)


def get_locale() -> LocaleInfo:
    """Shortcut for getting current locale."""
    return get_i18n_manager().get_current_locale()


class StreamlitI18n:
    """Streamlit integration for i18n."""
    
    @staticmethod
    def create_language_selector(
        manager: I18nManager,
        key: str = "language_selector"
    ) -> Optional[LocaleInfo]:
        """Create Streamlit language selector widget."""
        try:
            import streamlit as st
            
            locales = manager.list_locales()
            current = manager.get_current_locale()
            
            locale_codes = [l.code for l in locales]
            locale_names = [l.native_name for l in locales]
            
            current_index = locale_codes.index(current.code) if current else 0
            
            selected = st.selectbox(
                manager.translate("settings.language", "Language"),
                options=locale_codes,
                format_func=lambda x: locale_names[locale_codes.index(x)],
                index=current_index,
                key=key
            )
            
            if selected and selected != current.code:
                manager.set_locale(selected)
            
            return manager.get_current_locale()
            
        except ImportError:
            logger.warning("Streamlit not available for language selector")
            return None
    
    @staticmethod
    def apply_rtl_styles(locale: LocaleInfo) -> None:
        """Apply RTL styles if needed."""
        if locale.direction == "rtl":
            try:
                import streamlit as st
                st.markdown("""
                <style>
                    .stApp { direction: rtl; }
                    .stMarkdown, .stText { text-align: right; }
                </style>
                """, unsafe_allow_html=True)
            except ImportError:
                pass
