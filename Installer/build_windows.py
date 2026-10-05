"""
Build dist/LuminaSetup.exe: one windowed exe with the Lumina icon that asks
for administrator rights and carries everything a node needs.

    python Installer/build_windows.py      (needs pyinstaller; make_icon.py needs playwright + Pillow)

Linux needs no build: ship the repo (or a tarball of ml/, node/, exonet.onnx,
branding/, Installer/lumina_install.py) and run lumina_install.py with python3.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sep = ";"   # PyInstaller --add-data separator on Windows

if not (ROOT / "branding" / "lumina.ico").is_file():
    subprocess.run([sys.executable, str(ROOT / "Installer" / "make_icon.py")], check=True)

data = ["ml", "node", "exonet.onnx", "branding/lumina.ico", "branding/lumina-256.png",
        "Installer/python-embed", "Installer/get-pip.py"]
cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
       "--uac-admin", "--name", "LuminaSetup", "--icon", str(ROOT / "branding" / "lumina.ico"),
       "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build"), "--specpath", str(ROOT / "build"),
       "--exclude-module", "numpy", "--exclude-module", "torch"]
for d in data:
    dest = d if Path(d).suffix == "" else str(Path(d).parent)
    cmd += ["--add-data", f"{ROOT / d}{sep}{dest}"]
cmd.append(str(ROOT / "Installer" / "lumina_install.py"))
subprocess.run(cmd, check=True, cwd=ROOT)
print("built", ROOT / "dist" / "LuminaSetup.exe")
