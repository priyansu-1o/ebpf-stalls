import sys
from pathlib import Path

site_pkg = Path(__file__).resolve().parent / ".site-packages"
if site_pkg.exists() and str(site_pkg) not in sys.path:
    sys.path.insert(0, str(site_pkg))

