"""Main module for AI Flora Mind service"""


def hello_world() -> str:
    """Simple function to test the package."""
    return "Hello from AI Flora Mind!"


def get_version() -> str:
    """Get the package version."""
    from . import __version__
    return __version__