# Script to update CSS with new color scheme

# Read the file
with open('app/styles/custom_css.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Old colors to new colors mapping
replacements = {
    'NETFLIX_RED': 'ACCENT_BLUE',
    'NETFLIX_BLACK': 'PRIMARY_DARK',
    'NETFLIX_DARK_GRAY': 'SECONDARY_DARK',
    'NETFLIX_GRAY': 'SECONDARY_DARK',
    'NETFLIX_LIGHT_GRAY': 'LIGHT_COLOR',
    'SUCCESS_GREEN': 'ACCENT_BLUE',
    'WARNING_YELLOW': 'SECONDARY_DARK',
    'ERROR_RED': 'PRIMARY_DARK',
    'INFO_BLUE': 'ACCENT_BLUE',
    '#E50914': '#1D546C',
    '#141414': '#0C2B4E',
    '#222222': '#1A3D64',
    '#333333': '#1A3D64',
    '#757575': '#F4F4F4',
    '#4CAF50': '#1D546C',
    '#FFC107': '#1A3D64',
    '#F44336': '#0C2B4E',
    '#2196F3': '#1D546C',
    '#FF0A16': '#F4F4F4',
    '#CCC': '#F4F4F4',
    '#DDD': '#F4F4F4',
    '#FFD700': '#F4F4F4',
    'Netflix-inspired dark theme': 'New minimal blue color scheme',
    'Netflix Brand Colors': 'Project Brand Colors - New Color Scheme',
    'Netflix Red': 'Accent blue',
    'Dark background': 'Primary dark background',
}

# Apply replacements
for old, new in replacements.items():
    content = content.replace(old, new)

# Write back
with open('app/styles/custom_css.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Color scheme updated successfully!")
print("Old Netflix theme → New Blue theme")
print(f"#E50914 (Red) → #1D546C (Accent Blue)")
print(f"#141414 (Black) → #0C2B4E (Primary Dark)")
print(f"#222222/#333333 (Gray) → #1A3D64 (Secondary Dark)")
print(f"White/Light → #F4F4F4 (Light Color)")
