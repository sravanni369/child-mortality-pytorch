"""Capture a maximized VS Code window to PNG, cropped to hide the side bar.

Windows-only. Uses PrintWindow so it works without any typing into the IDE.
Run after launching:  code --disable-workspace-trust --new-window <folder>
"""

import ctypes
import sys
import time
from ctypes import wintypes

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


def find_vscode(title_must_contain=""):
    hwnds = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 256)
            if "Visual Studio Code" in buf.value and title_must_contain in buf.value:
                hwnds.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    return hwnds[0] if hwnds else None


def capture(hwnd, out_path, crop_right=0.80):
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    user32.ShowWindow(hwnd, 3)  # maximize
    user32.SetForegroundWindow(hwnd)
    time.sleep(1.5)

    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top

    hdc_win = user32.GetWindowDC(hwnd)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_win)
    bmp = gdi32.CreateCompatibleBitmap(hdc_win, w, h)
    gdi32.SelectObject(hdc_mem, bmp)
    user32.PrintWindow(hwnd, hdc_mem, 2)

    bih = BITMAPINFOHEADER(biSize=40, biWidth=w, biHeight=-h, biPlanes=1,
                           biBitCount=32, biCompression=0)
    buf = ctypes.create_string_buffer(w * h * 4)
    gdi32.GetDIBits(hdc_mem, bmp, 0, h, buf, ctypes.byref(bih), 0)

    from PIL import Image
    img = Image.frombuffer("RGB", (w, h), buf, "raw", "BGRX")
    img = img.crop((0, 0, int(w * crop_right), h))
    img.save(out_path)

    gdi32.DeleteObject(bmp)
    gdi32.DeleteDC(hdc_mem)
    user32.ReleaseDC(hwnd, hdc_win)
    print(f"saved {out_path} ({img.size[0]}x{img.size[1]})")


def close(title_must_contain):
    """Close a matching window. VS Code reuses an open window for the same
    folder, and the folderOpen task does not re-fire when it does - which
    silently screenshots the previous run's output."""
    hwnd = find_vscode(title_must_contain)
    if hwnd:
        user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
        print(f"closed existing window for {title_must_contain}")
    return hwnd is not None


if __name__ == "__main__":
    if "--close-only" in sys.argv:
        close(sys.argv[sys.argv.index("--close-only") + 1])
        sys.exit(0)

    hwnd = find_vscode(sys.argv[2] if len(sys.argv) > 2 else "")
    if not hwnd:
        sys.exit("no VS Code window found")
    capture(hwnd, sys.argv[1] if len(sys.argv) > 1 else "vscode_run.png")
