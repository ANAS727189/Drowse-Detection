from __future__ import annotations

import os

# Helps OpenCV Qt backend find fonts on Linux if they are present.
if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

from driver_monitor.app import main


if __name__ == "__main__":
    main()

