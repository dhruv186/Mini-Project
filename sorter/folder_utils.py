import os
import shutil
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def find_images(src_dir):
    for root, _, files in os.walk(src_dir):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                yield os.path.join(root, f)

#create hardlink/symlink for group photos
def create_link_to_file(target: Path, link_path: Path):
    target = Path(target)
    link_path = Path(link_path)
    if not target.exists():
        raise FileNotFoundError(f"Target does not exist: {target}")
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists():
        return
    # Try symlink
    try:
        os.symlink(str(target), str(link_path))
        return
    except OSError:
        pass
    # Try hard link 
    try:
        if target.is_file():
            os.link(str(target), str(link_path))
            return
    except OSError:
        pass

    #windows creating .lnk using win32com 
    if sys.platform.startswith("win"):
        try:
            from win32com.client import Dispatch
            link_with_lnk = link_path.with_suffix('.lnk')
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortcut(str(link_with_lnk))
            shortcut.TargetPath = str(target)
            shortcut.WorkingDirectory = str(target.parent)
            shortcut.save()
            return
        except Exception:
            pass
        #create a small .url file if link creation not possible
        try:
            url_path = link_path.with_suffix('.url')
            content = [
                "[InternetShortcut]",
                f"URL=file:///{str(target).replace(':','|')}", 
                f"Path={str(target)}",
            ]
            url_path.write_text("\n".join(content))
            return
        except Exception:
            pass
    
    if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
        try:
            desktop_path = link_path.with_suffix('.desktop')
            desktop_content = [
                "[Desktop Entry]",
                "Type=Link",
                f"Name={target.name}",
                f"Icon={shutil.which('xdg-icon-resource') or ''}",
                f"URL=file://{str(target)}",
                f"Comment=Shortcut to {target.name}"
            ]
            desktop_path.write_text("\n".join(desktop_content))
            desktop_path.chmod(0o755)
            return
        except Exception:
            pass

    # Error
    raise OSError(f"Failed to create symlink/hardlink/shortcut for {target} -> {link_path}")



def try_symlink_or_shortcut(target_path: Path, link_path: Path):
    try:
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        rel = os.path.relpath(target_path, link_path.parent)
        os.symlink(rel, link_path)
        return True
    except Exception:
        try:
            with open(str(link_path.with_suffix(link_path.suffix + ".shortcut.txt")), "w", encoding="utf-8") as fh:
                fh.write(str(target_path.resolve()))
            return False
        except Exception:
            return False

def copy_or_move(src: Path, dst: Path, move=False):
    ensure_dir(dst.parent)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))