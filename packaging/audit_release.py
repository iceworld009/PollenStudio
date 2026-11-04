"""Reject private/runtime files before producing a public source package."""
from pathlib import PurePosixPath
import re

ROOT_FILES = {
    '.gitattributes', '.gitignore', 'LICENSE', 'README.md', 'RELEASE_NOTES.md',
    'Setup-Linux.sh', 'Start-Linux.sh', 'Setup-Windows.cmd', 'Start-Windows.cmd',
    'capture.py', 'launch_studio.py', 'requirements.txt', 'requirements-training.txt',
    'start_studio.sh',
}
SOURCE_SUFFIXES = {'.py', '.js', '.html', '.css', '.md', '.txt', '.sh', '.ps1', '.cmd', '.yml', '.yaml'}
SOURCE_ROOTS = {'studio', 'tests', 'packaging', '.github'}


def validate_source(name, data):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe source path: '+name)
    if name not in ROOT_FILES and not (
        path.parts[0] in SOURCE_ROOTS and path.suffix in SOURCE_SUFFIXES
        and not any(part.startswith('.') and part != '.github' for part in path.parts)
        and '__pycache__' not in path.parts
    ):
        raise ValueError('File is not approved for public source packages: '+name)
    if len(data)>1024**2 or b'\0' in data:
        raise ValueError('Binary or unusually large source file: '+name)
    text=data.decode('utf-8')
    patterns=[
        r'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----',
        r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})',
        r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
        r'\bsk-(?:proj-)?[A-Za-z0-9_-]{25,}',
    ]
    if any(re.search(pattern,text) for pattern in patterns):
        raise ValueError('Possible credential in source file: '+name)
