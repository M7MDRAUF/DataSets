"""
Export Module for CineMatch V2.1.6

Provides functionality to export data in various formats
including CSV, JSON, and PDF.

Phase 6 - Task 6.3: Export Formats
"""

import csv
import io
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExportOptions:
    """Configuration options for exports."""
    include_headers: bool = True
    include_timestamps: bool = True
    include_metadata: bool = True
    date_format: str = "%Y-%m-%d %H:%M:%S"
    encoding: str = "utf-8"
    decimal_places: int = 2
    null_representation: str = ""
    # CSV specific
    csv_delimiter: str = ","
    csv_quotechar: str = '"'
    # JSON specific
    json_indent: int = 2
    json_ensure_ascii: bool = False
    # PDF specific
    pdf_title: Optional[str] = None
    pdf_author: str = "CineMatch"
    pdf_page_size: str = "A4"


class ExportFormat(ABC):
    """Abstract base class for export formats."""
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Get format name."""
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Get file extension."""
        pass
    
    @property
    @abstractmethod
    def mime_type(self) -> str:
        """Get MIME type."""
        pass
    
    @abstractmethod
    def export(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        options: Optional[ExportOptions] = None
    ) -> bytes:
        """
        Export data to bytes.
        
        Args:
            data: Data to export
            options: Export options
            
        Returns:
            Exported data as bytes
        """
        pass
    
    @abstractmethod
    def export_to_file(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        filepath: Path,
        options: Optional[ExportOptions] = None
    ) -> bool:
        """
        Export data to file.
        
        Args:
            data: Data to export
            filepath: Output file path
            options: Export options
            
        Returns:
            True if successful
        """
        pass


class CSVExporter(ExportFormat):
    """CSV format exporter."""
    
    @property
    def format_name(self) -> str:
        return "CSV"
    
    @property
    def file_extension(self) -> str:
        return ".csv"
    
    @property
    def mime_type(self) -> str:
        return "text/csv"
    
    def export(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        options: Optional[ExportOptions] = None
    ) -> bytes:
        """Export data to CSV bytes."""
        options = options or ExportOptions()
        
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Format datetime columns
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].dt.strftime(options.date_format)
        
        # Format float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].round(options.decimal_places)
        
        # Handle null values
        df = df.fillna(options.null_representation)
        
        # Export to bytes
        buffer = io.StringIO()
        df.to_csv(
            buffer,
            index=False,
            header=options.include_headers,
            sep=options.csv_delimiter,
            quotechar=options.csv_quotechar,
            encoding=options.encoding
        )
        
        return buffer.getvalue().encode(options.encoding)
    
    def export_to_file(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        filepath: Path,
        options: Optional[ExportOptions] = None
    ) -> bool:
        """Export data to CSV file."""
        try:
            content = self.export(data, options)
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(content)
            logger.info(f"Exported CSV to {filepath}")
            return True
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False


class JSONExporter(ExportFormat):
    """JSON format exporter."""
    
    @property
    def format_name(self) -> str:
        return "JSON"
    
    @property
    def file_extension(self) -> str:
        return ".json"
    
    @property
    def mime_type(self) -> str:
        return "application/json"
    
    def export(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        options: Optional[ExportOptions] = None
    ) -> bytes:
        """Export data to JSON bytes."""
        options = options or ExportOptions()
        
        # Convert DataFrame to records
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient='records')
        elif isinstance(data, dict):
            records = data
        else:
            records = data
        
        # Build export structure
        if options.include_metadata:
            export_data = {
                "metadata": {
                    "exported_at": datetime.utcnow().strftime(options.date_format),
                    "format": "JSON",
                    "source": "CineMatch",
                    "record_count": len(records) if isinstance(records, list) else 1
                },
                "data": records
            }
        else:
            export_data = records
        
        # Custom JSON encoder for datetime
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.strftime(options.date_format)
                if hasattr(obj, 'tolist'):  # numpy arrays
                    return obj.tolist()
                return super().default(obj)
        
        json_str = json.dumps(
            export_data,
            indent=options.json_indent,
            ensure_ascii=options.json_ensure_ascii,
            cls=DateTimeEncoder
        )
        
        return json_str.encode(options.encoding)
    
    def export_to_file(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        filepath: Path,
        options: Optional[ExportOptions] = None
    ) -> bool:
        """Export data to JSON file."""
        try:
            content = self.export(data, options)
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(content)
            logger.info(f"Exported JSON to {filepath}")
            return True
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False


class PDFExporter(ExportFormat):
    """PDF format exporter."""
    
    @property
    def format_name(self) -> str:
        return "PDF"
    
    @property
    def file_extension(self) -> str:
        return ".pdf"
    
    @property
    def mime_type(self) -> str:
        return "application/pdf"
    
    def export(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        options: Optional[ExportOptions] = None
    ) -> bytes:
        """Export data to PDF bytes."""
        options = options or ExportOptions()
        
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4, letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        except ImportError:
            # Fallback to simple HTML-based PDF
            return self._export_simple_pdf(data, options)
        
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Set page size
        page_sizes = {"A4": A4, "letter": letter}
        page_size = page_sizes.get(options.pdf_page_size, A4)
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=page_size,
            rightMargin=0.5*inch,
            leftMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title = options.pdf_title or "CineMatch Export"
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30
        )
        elements.append(Paragraph(title, title_style))
        
        # Metadata
        if options.include_metadata:
            meta_text = f"Exported: {datetime.utcnow().strftime(options.date_format)}<br/>Records: {len(df)}"
            elements.append(Paragraph(meta_text, styles['Normal']))
            elements.append(Spacer(1, 20))
        
        # Convert DataFrame to table data
        if options.include_headers:
            table_data = [list(df.columns)]
        else:
            table_data = []
        
        for _, row in df.iterrows():
            table_data.append([
                str(v)[:50] if v is not None else options.null_representation
                for v in row.values
            ])
        
        # Create table
        if table_data:
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(table)
        
        doc.build(elements)
        return buffer.getvalue()
    
    def _export_simple_pdf(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        options: ExportOptions
    ) -> bytes:
        """Simple PDF export without reportlab (HTML-based)."""
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="{options.encoding}">
            <title>{options.pdf_title or 'CineMatch Export'}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metadata {{ color: #666; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h1>{options.pdf_title or 'CineMatch Export'}</h1>
        """
        
        if options.include_metadata:
            html += f"""
            <div class="metadata">
                Exported: {datetime.utcnow().strftime(options.date_format)}<br>
                Records: {len(df)}
            </div>
            """
        
        html += df.to_html(index=False, na_rep=options.null_representation)
        html += "</body></html>"
        
        # Return HTML as bytes (would need wkhtmltopdf or similar for actual PDF)
        return html.encode(options.encoding)
    
    def export_to_file(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        filepath: Path,
        options: Optional[ExportOptions] = None
    ) -> bool:
        """Export data to PDF file."""
        try:
            content = self.export(data, options)
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(content)
            logger.info(f"Exported PDF to {filepath}")
            return True
        except Exception as e:
            logger.error(f"PDF export failed: {e}")
            return False


class ExcelExporter(ExportFormat):
    """Excel format exporter."""
    
    @property
    def format_name(self) -> str:
        return "Excel"
    
    @property
    def file_extension(self) -> str:
        return ".xlsx"
    
    @property
    def mime_type(self) -> str:
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    def export(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        options: Optional[ExportOptions] = None
    ) -> bytes:
        """Export data to Excel bytes."""
        options = options or ExportOptions()
        
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Format float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].round(options.decimal_places)
        
        buffer = io.BytesIO()
        
        try:
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(
                    writer,
                    sheet_name='Data',
                    index=False,
                    header=options.include_headers
                )
                
                if options.include_metadata:
                    meta_df = pd.DataFrame([
                        {"Property": "Exported At", "Value": datetime.utcnow().strftime(options.date_format)},
                        {"Property": "Records", "Value": len(df)},
                        {"Property": "Source", "Value": "CineMatch"}
                    ])
                    meta_df.to_excel(writer, sheet_name='Metadata', index=False)
        except ImportError:
            # Fallback to CSV if openpyxl not available
            csv_exporter = CSVExporter()
            return csv_exporter.export(df, options)
        
        return buffer.getvalue()
    
    def export_to_file(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        filepath: Path,
        options: Optional[ExportOptions] = None
    ) -> bool:
        """Export data to Excel file."""
        try:
            content = self.export(data, options)
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(content)
            logger.info(f"Exported Excel to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            return False


class ExportManager:
    """Manages data exports across formats."""
    
    _exporters: Dict[str, ExportFormat] = {}
    
    def __init__(self):
        # Register default exporters
        self.register_exporter(CSVExporter())
        self.register_exporter(JSONExporter())
        self.register_exporter(PDFExporter())
        self.register_exporter(ExcelExporter())
    
    def register_exporter(self, exporter: ExportFormat) -> None:
        """Register an exporter."""
        self._exporters[exporter.format_name.lower()] = exporter
        self._exporters[exporter.file_extension.lstrip('.')] = exporter
    
    def get_exporter(self, format: str) -> Optional[ExportFormat]:
        """Get exporter by format name or extension."""
        return self._exporters.get(format.lower())
    
    def list_formats(self) -> List[Dict[str, str]]:
        """List available export formats."""
        seen = set()
        formats = []
        for exporter in self._exporters.values():
            if exporter.format_name not in seen:
                seen.add(exporter.format_name)
                formats.append({
                    "name": exporter.format_name,
                    "extension": exporter.file_extension,
                    "mime_type": exporter.mime_type
                })
        return formats
    
    def export(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        format: str,
        options: Optional[ExportOptions] = None
    ) -> bytes:
        """Export data to specified format."""
        exporter = self.get_exporter(format)
        if not exporter:
            raise ValueError(f"Unknown export format: {format}")
        return exporter.export(data, options)
    
    def export_to_file(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        filepath: Union[str, Path],
        format: Optional[str] = None,
        options: Optional[ExportOptions] = None
    ) -> bool:
        """
        Export data to file.
        
        If format not specified, infers from file extension.
        """
        filepath = Path(filepath)
        
        # Infer format from extension if not specified
        if format is None:
            format = filepath.suffix.lstrip('.')
        
        exporter = self.get_exporter(format)
        if not exporter:
            raise ValueError(f"Unknown export format: {format}")
        
        return exporter.export_to_file(data, filepath, options)


# Specialized exporters for CineMatch data types

class RecommendationExporter:
    """Export recommendations in various formats."""
    
    def __init__(self, export_manager: Optional[ExportManager] = None):
        self._manager = export_manager or ExportManager()
    
    def export_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        format: str = "json",
        options: Optional[ExportOptions] = None
    ) -> bytes:
        """Export recommendations."""
        options = options or ExportOptions()
        options.pdf_title = options.pdf_title or "Movie Recommendations"
        
        # Ensure consistent structure
        formatted = []
        for rec in recommendations:
            formatted.append({
                "movie_id": rec.get("movie_id"),
                "title": rec.get("title", "Unknown"),
                "score": round(rec.get("score", 0), 3),
                "genres": ", ".join(rec.get("genres", [])) if isinstance(rec.get("genres"), list) else rec.get("genres", ""),
                "year": rec.get("year"),
                "explanation": rec.get("explanation", "")
            })
        
        return self._manager.export(formatted, format, options)
    
    def export_user_history(
        self,
        user_id: int,
        ratings: List[Dict[str, Any]],
        format: str = "json",
        options: Optional[ExportOptions] = None
    ) -> bytes:
        """Export user rating history."""
        options = options or ExportOptions()
        options.pdf_title = options.pdf_title or f"Rating History - User {user_id}"
        
        # Format ratings
        formatted = []
        for rating in ratings:
            formatted.append({
                "movie_id": rating.get("movie_id"),
                "title": rating.get("title", "Unknown"),
                "rating": rating.get("rating"),
                "rated_at": rating.get("timestamp", rating.get("rated_at", ""))
            })
        
        return self._manager.export(formatted, format, options)


class AnalyticsExporter:
    """Export analytics data."""
    
    def __init__(self, export_manager: Optional[ExportManager] = None):
        self._manager = export_manager or ExportManager()
    
    def export_stats(
        self,
        stats: Dict[str, Any],
        format: str = "json",
        options: Optional[ExportOptions] = None
    ) -> bytes:
        """Export analytics statistics."""
        options = options or ExportOptions()
        options.pdf_title = options.pdf_title or "Analytics Report"
        
        # Flatten nested stats
        flattened = self._flatten_dict(stats)
        
        if format.lower() == "json":
            return self._manager.export(stats, format, options)
        else:
            # For tabular formats, convert to list
            data = [{"metric": k, "value": v} for k, v in flattened.items()]
            return self._manager.export(data, format, options)
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '.'
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def export_performance_report(
        self,
        metrics: Dict[str, Any],
        format: str = "pdf",
        options: Optional[ExportOptions] = None
    ) -> bytes:
        """Export performance report."""
        options = options or ExportOptions()
        options.pdf_title = options.pdf_title or "Performance Report"
        
        data = []
        for algorithm, perf in metrics.items():
            if isinstance(perf, dict):
                row = {"algorithm": algorithm}
                row.update(perf)
                data.append(row)
        
        return self._manager.export(data, format, options)


# Global export manager instance
_export_manager: Optional[ExportManager] = None


def get_export_manager() -> ExportManager:
    """Get global export manager instance."""
    global _export_manager
    if _export_manager is None:
        _export_manager = ExportManager()
    return _export_manager


def export_data(
    data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
    format: str,
    options: Optional[ExportOptions] = None
) -> bytes:
    """Convenience function for exporting data."""
    return get_export_manager().export(data, format, options)


def export_to_file(
    data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
    filepath: Union[str, Path],
    format: Optional[str] = None,
    options: Optional[ExportOptions] = None
) -> bool:
    """Convenience function for exporting data to file."""
    return get_export_manager().export_to_file(data, filepath, format, options)
