#!/usr/bin/env python3
"""
Lumina node installer for Windows and Linux. Standard library only.

Turns this computer into a Lumina node: copies the worker, sets up a private
Python runtime with pinned packages, enrolls the machine with the coordinator
(getting its own revocable device token) and registers a background service
that starts at boot.

    LuminaSetup.exe                                  # Windows: installer window
    LuminaSetup.exe --quiet --enroll-token XYZ       # Windows: silent
    sudo python3 lumina_install.py [--quiet ...]     # Linux
    ... --uninstall [--purge]

Every option is a flag, so installs can be scripted; --quiet never prompts.
Re-running upgrades in place and keeps the existing enrollment. On Windows
all messages also go to %TEMP%\\lumina-install.log (the exe has no console).

Exit codes: 0 ok, 1 install failed, 2 bad arguments, 3 not elevated.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

WINDOWS = os.name == "nt"
FROZEN = getattr(sys, "frozen", False)
# Same layout in the repo and inside the packaged exe: ml/, node/, exonet.onnx,
# branding/, Installer/python-embed, Installer/get-pip.py
SOURCE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
DEFAULT_API = "https://lumina-exoplanet-hunter.onrender.com"
MISSION_CONTROL = "https://mbarc.github.io/Lumina-Exoplanet-Hunter/"
VERSION = "0.1.0"
SERVICE = "lumina-node"            # systemd unit (Linux)
TASK = "Lumina Node"               # scheduled task (Windows)
WIN_ACCOUNT = r"NT AUTHORITY\LOCALSERVICE"
WIN_SID = "*S-1-5-19"            # LocalService for icacls; names are localized on some Windows
ARP_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\Lumina"
LOG = Path(tempfile.gettempdir()) / "lumina-install.log"

QUIET = False
GUI_SINK = None                    # set by the installer window


class InstallError(Exception):
    pass


def asset(name: str) -> Path:
    """Icon from the release/exe, or next to the installed copy of this script."""
    for p in (SOURCE / "branding" / name, Path(__file__).resolve().parent / name):
        if p.is_file():
            return p
    return SOURCE / "branding" / name


def say(msg: str) -> None:
    if WINDOWS:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    if GUI_SINK:
        GUI_SINK(msg)
    elif not QUIET and sys.stdout:
        print(msg, flush=True)


def fail(msg: str) -> None:
    raise InstallError(msg)


def run(cmd: list[str], **kw) -> None:
    flags = subprocess.CREATE_NO_WINDOW if WINDOWS else 0
    res = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags, **kw)
    if res.returncode != 0:
        fail(f"{Path(cmd[0]).name} failed ({res.returncode}): {(res.stderr or res.stdout).strip()[-800:]}")


def powershell(script: str, check: bool = True) -> None:
    cmd = ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    if check:
        run(cmd)
    else:
        subprocess.run(cmd, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)


def ps_quote(s) -> str:
    return "'" + str(s).replace("'", "''") + "'"


# ── arguments ────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    if WINDOWS:
        install = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Lumina"
        data = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Lumina"
    else:
        install, data = Path("/opt/lumina"), Path("/var/lib/lumina")

    ap = argparse.ArgumentParser(prog="LuminaSetup" if FROZEN else None,
                                 description="Install, upgrade or remove a Lumina node.")
    ap.add_argument("--quiet", "-q", action="store_true", help="no window or prompts; errors only")
    ap.add_argument("--api-url", default=DEFAULT_API, help="coordinator URL (default: %(default)s)")
    ap.add_argument("--enroll-token", default=os.environ.get("LUMINA_ENROLL_TOKEN", ""),
                    help="enrollment token if the coordinator requires one (or set LUMINA_ENROLL_TOKEN)")
    ap.add_argument("--re-enroll", action="store_true", help="enroll again even if already enrolled")
    ap.add_argument("--install-dir", type=Path, default=install, help="program files (default: %(default)s)")
    ap.add_argument("--data-dir", type=Path, default=data, help="config, temp files (default: %(default)s)")
    ap.add_argument("--log-dir", type=Path, default=None, help="logs (default: <data-dir>/logs)")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="CPU threads the node may use (default: half the cores, %(default)s)")
    ap.add_argument("--batch-size", type=int, default=5, help="stars claimed per request (default: %(default)s)")
    ap.add_argument("--report-threshold", type=float, default=0.5,
                    help="send full light curves for scores at or above this (default: %(default)s)")
    ap.add_argument("--python", default=sys.executable,
                    help="Linux: interpreter for the node's venv (default: this one). "
                         "Windows always uses the bundled runtime.")
    ap.add_argument("--service-user", default="lumina", help="Linux account the service runs as (default: %(default)s)")
    ap.add_argument("--no-service", action="store_true", help="install files and enroll, but register no service")
    ap.add_argument("--no-start", action="store_true", help="register the service but don't start it now")
    ap.add_argument("--no-shortcuts", action="store_true", help="skip Start Menu / desktop entries")
    who = ap.add_argument_group("finder details (all optional; nothing is public unless you say so)")
    who.add_argument("--display-name", default=None, help="name shown on Mission Control if --public")
    who.add_argument("--credit-name", default=None,
                     help="name to credit if a candidate you find is submitted for review (with --credit)")
    who.add_argument("--email", default=None, help="for news about your candidates; never shown publicly")
    who.add_argument("--public", action=argparse.BooleanOptionalAction, default=None,
                     help="show (or with --no-public, stop showing) --display-name on Mission Control")
    who.add_argument("--credit", action=argparse.BooleanOptionalAction, default=None,
                     help="allow (or --no-credit, withdraw) --credit-name on candidate submissions "
                          "(needs a verified email)")
    ap.add_argument("--uninstall", action="store_true", help="remove the service and program files")
    ap.add_argument("--purge", action="store_true", help="with --uninstall, also delete data, config and logs")
    args = ap.parse_args(argv)
    args.log_dir = args.log_dir or args.data_dir / "logs"
    if args.threads < 1 or args.batch_size < 1:
        ap.error("--threads and --batch-size must be at least 1")
    if args.purge and not args.uninstall:
        ap.error("--purge only makes sense with --uninstall")
    return args


def is_elevated() -> bool:
    if WINDOWS:
        import ctypes
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    return os.geteuid() == 0


# ── install steps ────────────────────────────────────────────────────────────

def copy_program(args) -> None:
    app = args.install_dir / "app"
    say(f"Copying program files to {app}")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for pkg in ("ml", "node"):
        if not (SOURCE / pkg).is_dir():
            fail(f"missing {SOURCE / pkg}; run the installer from an unpacked Lumina release")
        shutil.copytree(SOURCE / pkg, app / pkg, ignore=ignore, dirs_exist_ok=True)
    if not (SOURCE / "exonet.onnx").is_file():
        fail(f"missing model {SOURCE / 'exonet.onnx'}")
    # Newer PyTorch exports keep the weights in exonet.onnx.data next to the graph.
    for f in ("exonet.onnx", "exonet.onnx.data"):
        if (SOURCE / f).is_file():
            shutil.copy2(SOURCE / f, app / f)
        else:
            (app / f).unlink(missing_ok=True)   # don't leave a stale weights file from an older model
    for icon in ("lumina.ico", "lumina-256.png"):
        shutil.copy2(SOURCE / "branding" / icon, args.install_dir / icon)
    # Keep a copy of the installer so Uninstall works after the download is gone.
    if FROZEN:
        shutil.copy2(sys.executable, args.install_dir / "LuminaSetup.exe")
    else:
        shutil.copy2(Path(__file__), args.install_dir / "lumina_install.py")
    if not WINDOWS:
        # Copies keep the source's mode bits; code run by the service must not
        # be writable by other users.
        run(["chmod", "-R", "go-w", str(args.install_dir)])


def build_runtime(args) -> Path:
    """Private Python with the node's packages. Returns the interpreter path."""
    req = str(args.install_dir / "app" / "node" / "requirements.txt")
    pip = ["-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-warn-script-location", "-r", req]
    if WINDOWS:
        # Bundled embeddable CPython: works from the frozen exe and needs nothing
        # preinstalled. Enable site-packages, bootstrap pip, then install.
        rt = args.install_dir / "python"
        say(f"Setting up the Lumina Python runtime in {rt}")
        shutil.copytree(SOURCE / "Installer" / "python-embed", rt, dirs_exist_ok=True)
        pth = next(rt.glob("python3*._pth"))
        # A ._pth file fixes sys.path exactly (no cwd), so list the app dir too.
        pth.write_text(f"{pth.stem}.zip\n.\nLib\\site-packages\n{args.install_dir / 'app'}\nimport site\n",
                       encoding="utf-8")
        py = rt / "python.exe"
        if not (rt / "Lib" / "site-packages" / "pip").is_dir():
            run([str(py), str(SOURCE / "Installer" / "get-pip.py"), "--quiet", "--no-warn-script-location"])
    else:
        venv = args.install_dir / "venv"
        say(f"Creating Python environment in {venv}")
        run([args.python, "-m", "venv", str(venv)])
        py = venv / "bin" / "python"
    say("Installing pinned packages (a few minutes on first install)")
    run([str(py), *pip])
    return py


def profile_from_args(args) -> dict | None:
    """Only the finder details actually given (flags / installer window), or None.

    The server applies them as a partial update, so leaving a flag out never
    wipes what is stored, and --no-public / --no-credit withdraw consent.
    """
    fields = {"display_name": args.display_name, "credit_name": args.credit_name, "email": args.email,
              "show_publicly": args.public, "credit_in_submissions": args.credit}
    given = {k: v for k, v in fields.items() if v is not None}
    return given or None


def api_call(args, method: str, path: str, body: dict | None = None, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Device-Token"] = token
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(args.api_url.rstrip("/") + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as r:   # free-tier hosts can take a while to wake
        return json.load(r)


def enroll(args, cfg_path: Path) -> dict:
    profile = profile_from_args(args)
    if cfg_path.is_file() and not args.re_enroll:
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old.get("device_token") and old.get("api_url") == args.api_url:
            say(f"Already enrolled as {old.get('device_name')}; keeping it")
            if profile is not None:
                say("Updating your finder details")
                try:
                    api_call(args, "PUT", "/nodes/me/profile", profile, token=old["device_token"])
                except (urllib.error.HTTPError, urllib.error.URLError) as e:
                    # Not worth failing an upgrade over; the node works without it.
                    say(f"WARNING: could not update finder details ({e}); everything else continues")
            return {k: old[k] for k in ("device_id", "device_name", "device_token")}

    token = args.enroll_token
    while True:
        say(f"Enrolling with {args.api_url}")
        body = {"hostname": socket.gethostname()[:64], "platform": platform.system(), "enroll_token": token}
        if profile is not None:
            body["profile"] = profile
        try:
            got = api_call(args, "POST", "/nodes/enroll", body)
            return {"device_id": got["device_id"], "device_name": got["name"], "device_token": got["device_token"]}
        except urllib.error.HTTPError as e:
            if e.code == 403 and not QUIET and not GUI_SINK and sys.stdin:
                token = getpass.getpass("This network needs an enrollment token: ").strip()
                if token:
                    continue
            if e.code == 403:
                fail("this network needs an enrollment token (--enroll-token), and the one given was wrong or missing")
            fail(f"enrollment rejected ({e.code}): {e.read().decode(errors='replace')[:300]}")
        except urllib.error.URLError as e:
            fail(f"cannot reach {args.api_url}: {e.reason}")


def write_config(args, cfg_path: Path, identity: dict) -> None:
    cfg = {
        "api_url": args.api_url, **identity,
        "model_path": str(args.install_dir / "app" / "exonet.onnx"),
        "data_dir": str(args.data_dir), "log_dir": str(args.log_dir),
        "threads": args.threads, "batch_size": args.batch_size,
        "report_threshold": args.report_threshold,
    }
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    lock_down(args, cfg_path)   # before the token touches disk
    fd = os.open(cfg_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(cfg, indent=2))
    if not WINDOWS:
        os.chmod(cfg_path, 0o600)   # an older file keeps its mode through O_CREAT
        run(["chown", f"{args.service_user}:", str(cfg_path)])
    say(f"Config written to {cfg_path}")


def lock_down(args, cfg_path: Path) -> None:
    """The device token is a credential: only the service account and admins may read
    its directory, and that is in place before the token is written."""
    cfg_dir = cfg_path.parent
    if WINDOWS:
        for d in {args.data_dir, args.log_dir}:   # a custom --log-dir needs its own grant
            run(["icacls", str(d), "/grant", f"{WIN_SID}:(OI)(CI)M", "/Q"])
        run(["icacls", str(cfg_dir), "/inheritance:r", "/grant:r", f"{WIN_SID}:(OI)(CI)R",
             "*S-1-5-32-544:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F", "/Q"])
        if cfg_path.exists():   # an older config: drop its own ACL, inherit the locked one
            run(["icacls", str(cfg_path), "/reset", "/Q"])
        return
    if subprocess.run(["id", "-u", args.service_user], capture_output=True).returncode != 0:
        nologin = "/usr/sbin/nologin" if Path("/usr/sbin/nologin").exists() else "/bin/false"
        run(["useradd", "--system", "--no-create-home", "--shell", nologin, args.service_user])
    os.chmod(cfg_dir, 0o700)
    for d in {args.data_dir, args.log_dir}:
        run(["chown", "-R", f"{args.service_user}:", str(d)])


def register_service(args, py: Path, cfg_path: Path) -> None:
    app = args.install_dir / "app"
    if WINDOWS:
        say(f"Registering background task '{TASK}' (starts with Windows, runs as LocalService)")
        pyw = py.with_name("pythonw.exe")
        task_args = '-m node.worker --config "' + str(cfg_path) + '"'
        powershell(
            f"$a = New-ScheduledTaskAction -Execute {ps_quote(pyw)} "
            f"-Argument {ps_quote(task_args)} -WorkingDirectory {ps_quote(app)}; "
            "$t = New-ScheduledTaskTrigger -AtStartup; "
            "$s = New-ScheduledTaskSettingsSet -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) "
            "-ExecutionTimeLimit (New-TimeSpan) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable; "
            f"$p = New-ScheduledTaskPrincipal -UserId {ps_quote(WIN_ACCOUNT)} -LogonType ServiceAccount; "
            f"Register-ScheduledTask -TaskName {ps_quote(TASK)} -Action $a -Trigger $t -Settings $s -Principal $p -Force | Out-Null"
        )
        if not args.no_start:
            powershell(f"Start-ScheduledTask -TaskName {ps_quote(TASK)}")
        return

    if not shutil.which("systemctl"):
        fail("systemd not found; rerun with --no-service and start the worker your own way")
    say(f"Registering systemd service {SERVICE}")
    Path(f"/etc/systemd/system/{SERVICE}.service").write_text(f"""\
[Unit]
Description=Lumina volunteer node (exoplanet search)
Wants=network-online.target
After=network-online.target

[Service]
User={args.service_user}
WorkingDirectory={app}
ExecStart={py} -m node.worker --config {cfg_path}
Restart=always
RestartSec=60
Nice=10
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={args.data_dir} {args.log_dir}

[Install]
WantedBy=multi-user.target
""", encoding="utf-8")
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "enable", SERVICE])
    if not args.no_start:
        run(["systemctl", "restart", SERVICE])


def uninstall_command(args, quiet: bool = False) -> tuple[str, str]:
    """(program, arguments) that run the installed copy of this installer with --uninstall."""
    # Custom locations must travel with the command, or uninstall removes the defaults.
    # list2cmdline gets Windows quoting right, e.g. a path ending in a backslash.
    argv = ["--uninstall", "--install-dir", str(args.install_dir), "--data-dir", str(args.data_dir),
            "--log-dir", str(args.log_dir)] + (["--quiet"] if quiet else [])
    if FROZEN:
        return str(args.install_dir / "LuminaSetup.exe"), subprocess.list2cmdline(argv)
    return (str(args.install_dir / "python" / "python.exe"),
            subprocess.list2cmdline([str(args.install_dir / "lumina_install.py"), *argv]))


def add_shortcuts(args) -> None:
    ico = args.install_dir / "lumina.ico"
    if WINDOWS:
        say("Adding Start Menu entries and Apps & Features entry")
        menu = Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Microsoft/Windows/Start Menu/Programs/Lumina"
        menu.mkdir(parents=True, exist_ok=True)
        (menu / "Lumina Mission Control.url").write_text(
            f"[InternetShortcut]\nURL={MISSION_CONTROL}\nIconFile={ico}\nIconIndex=0\n", encoding="utf-8")
        prog, prog_args = uninstall_command(args)
        links = [("Lumina logs", "explorer.exe", f'"{args.log_dir}"'), ("Uninstall Lumina", prog, prog_args)]
        script = "$s = New-Object -ComObject WScript.Shell; " + "".join(
            f"$l = $s.CreateShortcut({ps_quote(menu / (name + '.lnk'))}); $l.TargetPath = {ps_quote(target)}; "
            f"$l.Arguments = {ps_quote(a)}; $l.IconLocation = {ps_quote(str(ico) + ',0')}; $l.Save(); "
            for name, target, a in links)
        powershell(script)

        import winreg
        q_prog, q_args = uninstall_command(args, quiet=True)
        with winreg.CreateKeyEx(winreg.HKEY_LOCAL_MACHINE, ARP_KEY, 0, winreg.KEY_WRITE) as k:
            for name, value in {
                "DisplayName": "Lumina", "DisplayVersion": VERSION, "Publisher": "Lumina (independent project)",
                "DisplayIcon": str(ico), "InstallLocation": str(args.install_dir), "URLInfoAbout": MISSION_CONTROL,
                "UninstallString": f'"{prog}" {prog_args}', "QuietUninstallString": f'"{q_prog}" {q_args}',
            }.items():
                winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)
            for name in ("NoModify", "NoRepair"):
                winreg.SetValueEx(k, name, 0, winreg.REG_DWORD, 1)
        return

    apps = Path("/usr/share/applications")
    if apps.is_dir():
        say("Adding the Lumina icon and a Mission Control desktop entry")
        icon_dir = Path("/usr/share/icons/hicolor/256x256/apps")
        icon_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.install_dir / "lumina-256.png", icon_dir / "lumina.png")
        (apps / "lumina.desktop").write_text(
            f"[Desktop Entry]\nType=Link\nName=Lumina Mission Control\nURL={MISSION_CONTROL}\nIcon=lumina\n",
            encoding="utf-8")


# ── uninstall ────────────────────────────────────────────────────────────────

def stop_service() -> None:
    if WINDOWS:
        powershell(f"Stop-ScheduledTask -TaskName {ps_quote(TASK)} -ErrorAction SilentlyContinue", check=False)
    elif shutil.which("systemctl"):
        subprocess.run(["systemctl", "stop", SERVICE], capture_output=True)


MARKER = ".lumina-created"


def ours(d: Path) -> bool:
    """Lumina's own folder: created by the installer, or named for Lumina (the
    defaults incl. <data>/logs, and installs from before the marker existed)."""
    return (d / MARKER).exists() or "lumina" in d.name.lower() or "lumina" in d.parent.name.lower()


def claim_dirs(args) -> None:
    # Install chowns / re-ACLs these recursively and uninstall deletes them, so
    # a shared folder like D:\Data must never be one of them.
    for d in (args.install_dir, args.data_dir, args.log_dir):
        if not d.exists():
            d.mkdir(parents=True)
            (d / MARKER).touch()
        elif not ours(d):
            fail(f"{d} already exists and isn't a Lumina folder; pick a new folder (it will be created)")


def remove_tree(path: Path) -> None:
    if not ours(path):
        say(f"WARNING: {path} isn't a Lumina folder; left in place")
        return
    shutil.rmtree(path, ignore_errors=True)
    if path.exists() and WINDOWS:
        # Our own exe / interpreter lives here and is locked while we run:
        # delete the rest a few seconds after this process exits.
        subprocess.Popen(f'cmd /c ping -n 4 127.0.0.1 >nul & rmdir /s /q "{path}"',
                         creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW)


def uninstall(args) -> None:
    say("Removing Lumina node")
    stop_service()
    if WINDOWS:
        powershell(f"Unregister-ScheduledTask -TaskName {ps_quote(TASK)} -Confirm:$false -ErrorAction SilentlyContinue",
                   check=False)
        shutil.rmtree(Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Microsoft/Windows/Start Menu/Programs/Lumina",
                      ignore_errors=True)
        import winreg
        try:
            winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, ARP_KEY)
        except OSError:
            pass
    else:
        if shutil.which("systemctl"):
            subprocess.run(["systemctl", "disable", SERVICE], capture_output=True)
            Path(f"/etc/systemd/system/{SERVICE}.service").unlink(missing_ok=True)
            subprocess.run(["systemctl", "daemon-reload"], capture_output=True)
        Path("/usr/share/applications/lumina.desktop").unlink(missing_ok=True)
        Path("/usr/share/icons/hicolor/256x256/apps/lumina.png").unlink(missing_ok=True)
    if args.purge:
        remove_tree(args.log_dir)
        remove_tree(args.data_dir)
    remove_tree(args.install_dir)
    if args.purge:
        say("Removed Lumina, its data and logs. (The device stays enrolled until an operator revokes it.)")
    else:
        say(f"Removed Lumina. Data and config kept in {args.data_dir} (use --purge to delete).")


# ── orchestration ────────────────────────────────────────────────────────────

def check_python(args) -> None:
    """Linux: the venv interpreter must be usable before we stop a working node."""
    if WINDOWS:
        return   # bundled runtime
    probe = "import sys, venv, ensurepip; sys.exit(sys.version_info < (3, 12))"
    try:
        ok = subprocess.run([args.python, "-c", probe], capture_output=True).returncode == 0
    except OSError:
        ok = False
    if not ok:
        fail(f"{args.python} must be Python 3.12+ with venv and ensurepip "
             "(Debian/Ubuntu: apt install python3-venv); pick another with --python")


def install(args) -> str:
    cfg_path = args.data_dir / "config" / "config.json"
    check_python(args)
    claim_dirs(args)
    stop_service()   # upgrade in place: release file locks before copying
    copy_program(args)
    py = build_runtime(args)
    identity = enroll(args, cfg_path)
    write_config(args, cfg_path, identity)
    if not args.no_service:
        register_service(args, py, cfg_path)
    if not args.no_shortcuts:
        add_shortcuts(args)
    say(f"Done. This computer is {identity['device_name']} on the Lumina network.")
    return identity["device_name"]


def execute(args) -> None:
    if sys.version_info < (3, 10):
        fail("Python 3.10 or newer is required")
    if args.uninstall:
        uninstall(args)
    else:
        install(args)


def gui(args) -> None:
    """Windows installer window: the logo in the window and taskbar, a couple of
    choices, and a live log. Everything else uses the command-line defaults."""
    import ctypes
    import queue
    import tkinter as tk
    from tkinter import ttk

    global GUI_SINK
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Lumina.Installer")
    bg, fg, accent, muted = "#04090f", "#e8f4ff", "#00c8ff", "#8aa4c2"
    root = tk.Tk()
    root.title("Lumina Setup")
    root.iconbitmap(default=str(asset("lumina.ico")))
    root.configure(bg=bg, padx=28, pady=22)
    root.resizable(False, False)

    logo = tk.PhotoImage(file=str(asset("lumina-256.png"))).subsample(2)
    tk.Label(root, image=logo, bg=bg).grid(row=0, column=0, rowspan=3, padx=(0, 22))
    tk.Label(root, text="Lumina", font=("Segoe UI", 22, "bold"), fg=fg, bg=bg).grid(row=0, column=1, sticky="sw")
    verb = "Remove" if args.uninstall else "Install"
    blurb = ("Remove the Lumina node from this computer." if args.uninstall else
             "Lend this computer's spare time to the search for planets\naround other stars. "
             "It runs quietly in the background.")
    tk.Label(root, text=blurb, font=("Segoe UI", 10), fg=muted, bg=bg, justify="left").grid(row=1, column=1, sticky="nw")

    form = tk.Frame(root, bg=bg)
    form.grid(row=2, column=1, sticky="w", pady=(10, 0))
    token = tk.StringVar(value=args.enroll_token)
    threads = tk.IntVar(value=args.threads)
    if not args.uninstall:
        tk.Label(form, text="Enrollment token (only if you were given one)", fg=fg, bg=bg,
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        ttk.Entry(form, textvariable=token, width=36, show="•").grid(row=1, column=0, sticky="w", pady=(2, 8))
        tk.Label(form, text=f"CPU threads to share (of {os.cpu_count()})", fg=fg, bg=bg,
                 font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w")
        ttk.Spinbox(form, from_=1, to=os.cpu_count() or 1, textvariable=threads, width=6).grid(row=3, column=0, sticky="w")

    # Optional finder details. Empty and unticked by default: nothing is shared
    # unless the volunteer chooses to.
    # On an upgrade, start from the volunteer's current choices so the
    # always-sent checkboxes never withdraw consent by accident.
    current: dict = {}
    cfg_path = args.data_dir / "config" / "config.json"
    if not args.uninstall and cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if cfg.get("api_url") == args.api_url:
                current = api_call(args, "GET", "/nodes/me/profile", token=cfg["device_token"])
        except Exception:
            pass   # unreadable config or offline: fall back to empty fields
    who = {k: tk.StringVar(value=getattr(args, k) or current.get(k) or "")
           for k in ("display_name", "credit_name", "email")}
    public = tk.BooleanVar(value=args.public if args.public is not None else bool(current.get("show_publicly")))
    credit = tk.BooleanVar(value=args.credit if args.credit is not None
                           else bool(current.get("credit_in_submissions")))
    if not args.uninstall:
        about = tk.LabelFrame(root, text=" About you (optional) ", fg=accent, bg=bg, font=("Segoe UI", 9),
                              padx=10, pady=6)
        about.grid(row=3, column=0, columnspan=2, sticky="we", pady=(14, 0))
        for r, (key, label) in enumerate((("display_name", "Display name"),
                                          ("credit_name", "Name to credit on a discovery"),
                                          ("email", "Email (never shown; for news about your finds)"))):
            tk.Label(about, text=label, fg=fg, bg=bg, font=("Segoe UI", 9)).grid(row=r, column=0, sticky="w", padx=(0, 10))
            ttk.Entry(about, textvariable=who[key], width=40).grid(row=r, column=1, sticky="w", pady=2)
        tk.Checkbutton(about, text="Show my display name on Mission Control", variable=public, fg=fg, bg=bg,
                       selectcolor="#0d1f3c", activebackground=bg, activeforeground=fg).grid(
            row=3, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(about, text="Credit my name if a candidate I find is submitted for review",
                       variable=credit, fg=fg, bg=bg, selectcolor="#0d1f3c", activebackground=bg,
                       activeforeground=fg).grid(row=4, column=0, columnspan=2, sticky="w")

    log = tk.Text(root, height=8, width=78, bg="#0d1f3c", fg=fg, relief="flat", font=("Consolas", 9),
                  state="disabled", padx=8, pady=6)
    log.grid(row=4, column=0, columnspan=2, pady=(14, 12))
    buttons = tk.Frame(root, bg=bg)
    buttons.grid(row=5, column=0, columnspan=2, sticky="e")
    go = ttk.Button(buttons, text=verb)
    close = ttk.Button(buttons, text="Cancel", command=root.destroy)
    go.pack(side="left", padx=6)
    close.pack(side="left")

    lines: queue.Queue = queue.Queue()
    GUI_SINK = lines.put

    def pump():
        while not lines.empty():
            log.configure(state="normal")
            log.insert("end", lines.get() + "\n")
            log.see("end")
            log.configure(state="disabled")
        root.after(150, pump)

    def worker():
        try:
            execute(args)
            lines.put("")
            lines.put("All set. You can close this window." if not args.uninstall else "Lumina has been removed.")
        except InstallError as e:
            lines.put(f"ERROR: {e}")
            lines.put(f"Details: {LOG}")
            root.after(0, lambda: go.state(["!disabled"]))
        root.after(0, lambda: close.configure(text="Close"))

    def start():
        if not is_elevated():
            lines.put("ERROR: Lumina Setup needs administrator rights. Right-click it and choose Run as administrator.")
            return
        args.enroll_token, args.threads = token.get().strip(), max(1, int(threads.get()))
        # Typed fields are sent only if filled in (an empty box never wipes a
        # stored name); the two consents are always sent, so unticking opts out.
        for key, var in who.items():
            if var.get().strip():
                setattr(args, key, var.get().strip())
        args.public, args.credit = public.get(), credit.get()
        go.state(["disabled"])
        threading.Thread(target=worker, daemon=True).start()

    go.configure(command=start)
    pump()
    root.mainloop()


def main(argv=None) -> None:
    global QUIET
    args = parse_args(argv)
    QUIET = args.quiet
    if WINDOWS and not QUIET:
        gui(args)
        return
    if not is_elevated():
        msg = "run as Administrator (Windows) or root (Linux)"
        say(f"ERROR: {msg}")
        if sys.stderr:
            print(f"lumina-install: error: {msg}", file=sys.stderr)
        sys.exit(3)
    if not QUIET and not args.uninstall:
        say(f"Lumina node install\n  coordinator: {args.api_url}\n  program:     {args.install_dir}\n"
            f"  data:        {args.data_dir}\n  threads:     {args.threads}\n"
            f"  service:     {'none' if args.no_service else SERVICE}")
        if input("Proceed? [Y/n] ").strip().lower() not in ("", "y", "yes"):
            sys.exit(1)
    try:
        execute(args)
    except InstallError as e:
        say(f"ERROR: {e}")
        if sys.stderr:
            print(f"lumina-install: error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
