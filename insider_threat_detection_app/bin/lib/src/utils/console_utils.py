"""
Console utilities for Unicode-safe output on Windows and other platforms.
Handles encoding issues with emoji characters in Windows Command Prompt.
"""

import sys
import os
import subprocess
from typing import Any, Optional


def safe_print(*args, **kwargs) -> None:
    """
    Unicode-safe print function that handles encoding issues on Windows.
    Falls back to ASCII equivalents for emoji characters if needed.
    """
    # Try to print with UTF-8 encoding first
    try:
        # Force UTF-8 encoding for stdout if possible
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except (AttributeError, OSError):
                pass
        
        print(*args, **kwargs)
        
    except UnicodeEncodeError:
        # Fallback: replace Unicode emojis with ASCII equivalents
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                # Replace common emoji characters with ASCII equivalents
                safe_arg = (arg
                    .replace('[PASS]', '[PASS]')
                    .replace('[FAIL]', '[FAIL]')
                    .replace('[TEST]', '[TEST]')
                    .replace('[START]', '[START]')
                    .replace('[STATS]', '[STATS]')
                    .replace('[SUCCESS]', '[SUCCESS]')
                    .replace('[WARN]', '[WARN]')
                    .replace('[SCORE]', '[SCORE]')
                    .replace('[SETUP]', '[SETUP]')
                    .replace('[SAVE]', '[SAVE]')
                    .replace('[SEARCH]', '[SEARCH]')
                    .replace('[TIME]', '[TIME]')
                    .replace('[FINISH]', '[FINISH]')
                )
                safe_args.append(safe_arg)
            else:
                safe_args.append(arg)
        
        print(*safe_args, **kwargs)


def safe_subprocess_run(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """
    Unicode-safe subprocess.run that handles encoding issues.
    Sets appropriate environment variables for UTF-8 support.
    """
    # Set environment variables for UTF-8 support
    env = kwargs.get('env', os.environ.copy())
    
    # Force UTF-8 encoding on Windows
    if sys.platform.startswith('win'):
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
    
    kwargs['env'] = env
    
    # Ensure text mode and UTF-8 encoding
    if 'encoding' not in kwargs:
        kwargs['encoding'] = 'utf-8'
    
    if 'errors' not in kwargs:
        kwargs['errors'] = 'replace'  # Replace problematic characters instead of failing
    
    try:
        return subprocess.run(cmd, **kwargs)
    except UnicodeDecodeError:
        # Fallback with different encoding
        kwargs['encoding'] = 'cp1252'
        kwargs['errors'] = 'replace'
        return subprocess.run(cmd, **kwargs)


def setup_console_encoding() -> None:
    """
    Setup console encoding for better Unicode support.
    Call this at the start of scripts that use Unicode characters.
    """
    if sys.platform.startswith('win'):
        try:
            # Try to set console to UTF-8 mode on Windows
            os.system('chcp 65001 >nul 2>&1')
        except:
            pass
        
        # Set environment variables
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        os.environ['PYTHONUTF8'] = '1'
    
    # Try to reconfigure stdout/stderr for UTF-8
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass


def get_safe_emoji(emoji: str, fallback: str) -> str:
    """
    Get emoji if supported, otherwise return fallback.
    
    Args:
        emoji: Unicode emoji character
        fallback: ASCII fallback string
        
    Returns:
        Emoji if supported, fallback otherwise
    """
    try:
        # Test if we can encode the emoji
        emoji.encode(sys.stdout.encoding or 'utf-8')
        return emoji
    except (UnicodeEncodeError, LookupError):
        return fallback


# Predefined safe emoji mappings
SAFE_EMOJIS = {
    'check': get_safe_emoji('[PASS]', '[PASS]'),
    'cross': get_safe_emoji('[FAIL]', '[FAIL]'),
    'test': get_safe_emoji('[TEST]', '[TEST]'),
    'rocket': get_safe_emoji('[START]', '[START]'),
    'chart': get_safe_emoji('[STATS]', '[STATS]'),
    'party': get_safe_emoji('[SUCCESS]', '[SUCCESS]'),
    'warning': get_safe_emoji('[WARN]', '[WARN]'),
    'trending': get_safe_emoji('[SCORE]', '[SCORE]'),
    'wrench': get_safe_emoji('[SETUP]', '[SETUP]'),
    'floppy': get_safe_emoji('[SAVE]', '[SAVE]'),
    'mag': get_safe_emoji('[SEARCH]', '[SEARCH]'),
    'stopwatch': get_safe_emoji('[TIME]', '[TIME]'),
    'checkered_flag': get_safe_emoji('[FINISH]', '[FINISH]'),
}


# Convenience functions for common patterns
def print_success(message: str) -> None:
    """Print a success message with appropriate emoji."""
    safe_print(f"{SAFE_EMOJIS['check']} {message}")


def print_error(message: str) -> None:
    """Print an error message with appropriate emoji."""
    safe_print(f"{SAFE_EMOJIS['cross']} {message}")


def print_test(message: str) -> None:
    """Print a test message with appropriate emoji."""
    safe_print(f"{SAFE_EMOJIS['test']} {message}")


def print_header(message: str) -> None:
    """Print a header message with appropriate emoji."""
    safe_print(f"{SAFE_EMOJIS['rocket']} {message}")


def print_stats(message: str) -> None:
    """Print a statistics message with appropriate emoji."""
    safe_print(f"{SAFE_EMOJIS['chart']} {message}")


def print_warning(message: str) -> None:
    """Print a warning message with appropriate emoji."""
    safe_print(f"{SAFE_EMOJIS['warning']} {message}")
