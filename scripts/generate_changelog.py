#!/usr/bin/env python3
"""
CineMatch V2.1.6 - Changelog Generator

Generates CHANGELOG.md from git commits following Conventional Commits specification.
Parses commit messages and organizes them by type and scope.

Usage:
    python scripts/generate_changelog.py [options]
    
Options:
    --from-tag TAG    Start from this git tag (default: previous tag)
    --to-tag TAG      End at this tag (default: HEAD)
    --output FILE     Output file path (default: CHANGELOG.md)
    --version VER     Version number for release (default: from VERSION file)
    --dry-run         Print to stdout instead of file

Author: CineMatch Development Team
Date: December 2025
"""

import subprocess
import re
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


@dataclass
class Commit:
    """Represents a parsed git commit."""
    
    hash: str
    type: str
    scope: Optional[str]
    description: str
    body: str
    breaking: bool
    footer: Dict[str, str] = field(default_factory=dict)
    
    @property
    def short_hash(self) -> str:
        return self.hash[:7]
    
    def format_entry(self) -> str:
        """Format commit as changelog entry."""
        scope_str = f"**{self.scope}:** " if self.scope else ""
        breaking_str = "⚠️ BREAKING: " if self.breaking else ""
        return f"- {breaking_str}{scope_str}{self.description} ({self.short_hash})"


# Conventional Commits regex pattern
COMMIT_PATTERN = re.compile(
    r'^(?P<type>feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)'
    r'(?:\((?P<scope>[^)]+)\))?'
    r'(?P<breaking>!)?'
    r':\s*(?P<description>.+)$',
    re.MULTILINE
)

# Commit type to changelog section mapping
TYPE_MAPPING = {
    'feat': ('✨ Features', 1),
    'fix': ('🐛 Bug Fixes', 2),
    'perf': ('⚡ Performance', 3),
    'refactor': ('♻️ Refactoring', 4),
    'docs': ('📚 Documentation', 5),
    'test': ('✅ Tests', 6),
    'build': ('🔧 Build System', 7),
    'ci': ('👷 CI/CD', 8),
    'style': ('💄 Style', 9),
    'chore': ('🔨 Chores', 10),
    'revert': ('⏪ Reverts', 11),
}


def run_git_command(args: List[str]) -> str:
    """Execute a git command and return output."""
    result = subprocess.run(
        ['git'] + args,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    if result.returncode != 0:
        print(f"Git error: {result.stderr}", file=sys.stderr)
        return ""
    return result.stdout.strip()


def get_tags() -> List[str]:
    """Get list of git tags sorted by version."""
    output = run_git_command(['tag', '--sort=-v:refname'])
    return output.split('\n') if output else []


def get_commits(from_ref: str, to_ref: str = 'HEAD') -> List[Dict[str, str]]:
    """Get commits between two refs."""
    format_str = '%H%n%s%n%b%n---COMMIT_END---'
    
    if from_ref:
        range_str = f'{from_ref}..{to_ref}'
    else:
        range_str = to_ref
    
    output = run_git_command([
        'log',
        range_str,
        f'--format={format_str}'
    ])
    
    commits = []
    for commit_text in output.split('---COMMIT_END---'):
        commit_text = commit_text.strip()
        if not commit_text:
            continue
        
        lines = commit_text.split('\n')
        if len(lines) >= 2:
            commits.append({
                'hash': lines[0],
                'subject': lines[1],
                'body': '\n'.join(lines[2:]) if len(lines) > 2 else ''
            })
    
    return commits


def parse_commit(raw: Dict[str, str]) -> Optional[Commit]:
    """Parse a raw commit into a Commit object."""
    subject = raw['subject']
    match = COMMIT_PATTERN.match(subject)
    
    if not match:
        # Non-conventional commit, categorize as chore
        return Commit(
            hash=raw['hash'],
            type='chore',
            scope=None,
            description=subject,
            body=raw['body'],
            breaking=False
        )
    
    breaking = bool(match.group('breaking'))
    body = raw['body']
    
    # Check for BREAKING CHANGE in body
    if 'BREAKING CHANGE:' in body or 'BREAKING-CHANGE:' in body:
        breaking = True
    
    return Commit(
        hash=raw['hash'],
        type=match.group('type'),
        scope=match.group('scope'),
        description=match.group('description'),
        body=body,
        breaking=breaking
    )


def group_commits(commits: List[Commit]) -> Dict[str, List[Commit]]:
    """Group commits by type."""
    grouped = defaultdict(list)
    
    for commit in commits:
        grouped[commit.type].append(commit)
    
    return grouped


def generate_changelog(
    commits: List[Commit],
    version: str,
    date: str = None
) -> str:
    """Generate changelog markdown content."""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    lines = []
    
    # Version header
    lines.append(f"## [{version}] - {date}")
    lines.append("")
    
    # Group by type
    grouped = group_commits(commits)
    
    # Breaking changes section first
    breaking_commits = [c for c in commits if c.breaking]
    if breaking_commits:
        lines.append("### ⚠️ Breaking Changes")
        lines.append("")
        for commit in breaking_commits:
            lines.append(commit.format_entry())
        lines.append("")
    
    # Other sections sorted by priority
    sorted_types = sorted(
        grouped.keys(),
        key=lambda t: TYPE_MAPPING.get(t, ('', 99))[1]
    )
    
    for commit_type in sorted_types:
        if commit_type not in TYPE_MAPPING:
            continue
        
        section_name, _ = TYPE_MAPPING[commit_type]
        type_commits = [c for c in grouped[commit_type] if not c.breaking]
        
        if not type_commits:
            continue
        
        lines.append(f"### {section_name}")
        lines.append("")
        
        # Group by scope within type
        scoped = defaultdict(list)
        for commit in type_commits:
            scope = commit.scope or ''
            scoped[scope].append(commit)
        
        for scope in sorted(scoped.keys()):
            for commit in scoped[scope]:
                lines.append(commit.format_entry())
        
        lines.append("")
    
    return '\n'.join(lines)


def read_existing_changelog(filepath: Path) -> str:
    """Read existing changelog content."""
    if filepath.exists():
        return filepath.read_text(encoding='utf-8')
    return ""


def merge_changelog(existing: str, new_content: str) -> str:
    """Merge new changelog content with existing."""
    header = """# Changelog

All notable changes to CineMatch will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

"""
    
    # Find where existing releases start
    if existing:
        # Remove existing header
        if existing.startswith('# Changelog'):
            idx = existing.find('## [')
            if idx > 0:
                existing = existing[idx:]
        
        return header + new_content + "\n" + existing
    
    return header + new_content


def get_version_from_file() -> str:
    """Read version from VERSION file."""
    version_file = Path(__file__).parent.parent / 'VERSION'
    if version_file.exists():
        return version_file.read_text().strip()
    return "0.0.0"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate changelog from git commits')
    parser.add_argument('--from-tag', help='Start from this git tag')
    parser.add_argument('--to-tag', default='HEAD', help='End at this tag')
    parser.add_argument('--output', default='CHANGELOG.md', help='Output file path')
    parser.add_argument('--version', help='Version number for release')
    parser.add_argument('--dry-run', action='store_true', help='Print to stdout')
    
    args = parser.parse_args()
    
    # Determine version
    version = args.version or get_version_from_file()
    
    # Determine from tag
    from_tag = args.from_tag
    if not from_tag:
        tags = get_tags()
        if tags:
            from_tag = tags[0]
    
    print(f"Generating changelog for version {version}")
    if from_tag:
        print(f"From: {from_tag} to: {args.to_tag}")
    else:
        print(f"All commits up to: {args.to_tag}")
    
    # Get and parse commits
    raw_commits = get_commits(from_tag, args.to_tag)
    print(f"Found {len(raw_commits)} commits")
    
    if not raw_commits:
        print("No commits found")
        return
    
    commits = []
    for raw in raw_commits:
        commit = parse_commit(raw)
        if commit:
            commits.append(commit)
    
    # Generate changelog content
    new_content = generate_changelog(commits, version)
    
    if args.dry_run:
        print("\n" + "="*60)
        print(new_content)
        return
    
    # Read existing and merge
    output_path = Path(args.output)
    existing = read_existing_changelog(output_path)
    final_content = merge_changelog(existing, new_content)
    
    # Write output
    output_path.write_text(final_content, encoding='utf-8')
    print(f"Changelog written to {output_path}")


if __name__ == '__main__':
    main()
