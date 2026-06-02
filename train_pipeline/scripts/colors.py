"""ANSI color helpers for terminal output."""

import os
import sys

# Auto-detect: disable colors if not a TTY, unless FORCE_COLOR is set
_USE_COLOR = (
    os.environ.get('FORCE_COLOR', '') == '1'
    or (hasattr(sys.stdout, 'isatty') and sys.stdout.isatty())
)


def _wrap(code, text):
    if not _USE_COLOR:
        return text
    return f'\033[{code}m{text}\033[0m'


# ── Basic colors ──
def red(t):      return _wrap('31', t)
def green(t):    return _wrap('32', t)
def yellow(t):   return _wrap('33', t)
def blue(t):     return _wrap('34', t)
def magenta(t):  return _wrap('35', t)
def cyan(t):     return _wrap('36', t)
def white(t):    return _wrap('37', t)
def gray(t):     return _wrap('90', t)

# ── Bold ──
def bold(t):     return _wrap('1', t)
def bold_green(t):  return _wrap('1;32', t)
def bold_red(t):    return _wrap('1;31', t)
def bold_cyan(t):   return _wrap('1;36', t)
def bold_yellow(t): return _wrap('1;33', t)
def bold_magenta(t): return _wrap('1;35', t)
def bold_blue(t):   return _wrap('1;34', t)


def enable_color(force=None):
    """Force color on/off. If force=None, auto-detect from TTY."""
    global _USE_COLOR
    if force is not None:
        _USE_COLOR = force
    else:
        _USE_COLOR = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
