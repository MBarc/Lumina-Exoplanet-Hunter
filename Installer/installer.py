import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import platform
import os
import json
import subprocess
import threading
import pymongo
import urllib.request
import zipfile
import tarfile
import sys

def resource_path(relative_path: str) -> str:
    """Get absolute path to resource (works in dev and PyInstaller)"""
    if hasattr(sys, "_MEIPASS"):
        # Running inside PyInstaller bundle
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)

class ExoplanetHunterApp(tk.Tk):

    FG_COLOR = "white"
    BG_COLOR = "#0b1a33"
    BACK_BUTTON_TEXT = "⬅️ Back"
    NEXT_BUTTON_TEXT = "Next ➡️"

    def __init__(self):
        super().__init__()
        self.title("Exoplanet Hunter Worker v1.0.0")
        self.geometry("900x600")
        self.resizable(False, False)
        self.container = tk.Frame(self, bg="#0b1a33")
        self.container.pack(fill="both", expand=True)

        # Custom button style
        style = ttk.Style(self)
        style.configure(
            "Custom.TButton",
            background=self.BG_COLOR,
            foreground="black",
            padding=10,
            borderwidth=0,
            focusthickness=3,
            focuscolor="none"
        )
        style.map(
            "Custom.TButton",
            background=[("active", "#2a4d9c")]  # lighter blue on hover
        )

        style.configure(
            "Browse.TButton",
            background=self.BG_COLOR,
            foreground="black",
            padding=(10, 2),
            borderwidth=0,
            focusthickness=3,
            focuscolor="none"
        )
        style.map(
            "Browse.TButton",
            background=[("active", "#2a4d9c")]  # lighter blue on hover
        )

        self.frames = {}
        for F in (WelcomePage, InstallDirPage, DBCredentialsPage, DBConnectionPage, FeaturesPage, SummaryPage, InstallationPage, CompletionPage):
            frame = F(parent=self.container, controller=self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(WelcomePage)

    def show_frame(self, page_class):
        frame = self.frames[page_class]
        frame.tkraise()
        if page_class == SummaryPage:
            frame.update_summary()
        if page_class == InstallationPage:
            frame.start_installation()

class WelcomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller

        # Configure grid to center content both vertically and horizontally
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(2, weight=1)

        # Create centered content frame
        content = tk.Frame(self, bg=self.controller.BG_COLOR)
        content.grid(row=1, column=1, sticky="nsew")

        title = tk.Label(
            content,
            text="🌍 Welcome to Exoplanet Hunter! 🌍",
            font=("Helvetica", 20, "bold"),
            fg=self.controller.FG_COLOR,
            bg=self.controller.BG_COLOR
        )
        title.pack(pady=20)

        message = (
            "Join the exciting search for exoplanets using NASA's TESS satellite data!\n\n"
            "This worker node will contribute to analyzing light curve data from over 1 million stars,\n"
            "helping scientists discover new worlds beyond our solar system. Your computational\n"
            "resources will be part of a distributed network processing cutting-edge astronomical data.\n\n"
            "Together, we're pushing the boundaries of space exploration and expanding our\n"
            "understanding of planetary systems throughout the galaxy.\n\n"
            "Ready to contribute to this groundbreaking mission?"
        )

        label = tk.Label(
            content,
            text=message,
            justify="center",   # <-- changed from left to center
            wraplength=650,
            font=("Helvetica", 12),
            fg=self.controller.FG_COLOR,
            bg=self.controller.BG_COLOR
        )
        label.pack(padx=20, pady=20)

        next_button = ttk.Button(content, text="I'm ready!", command=lambda: controller.show_frame(InstallDirPage), style="Custom.TButton")
        next_button.pack(pady=20)



class InstallDirPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller

        # Centering grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        content = tk.Frame(self, bg=self.controller.BG_COLOR)
        content.grid(row=1, column=0)

        title = tk.Label(content, text="📂 Installation Directories 📂", 
                         font=("Helvetica", 18, "bold"), fg=self.controller.FG_COLOR, bg=self.controller.BG_COLOR)
        title.pack(pady=20)

        system = platform.system()
        if system == "Windows":
            default_data = "C:\\Program Files\\ExoplanetHunter\\Data"
            default_logs = "C:\\Program Files\\ExoplanetHunter\\Logs"
        else:
            default_data = "/var/lib/exoplanet_hunter/data"
            default_logs = "/var/log/exoplanet_hunter"

        data_frame = tk.Frame(content, bg=self.controller.BG_COLOR)
        data_frame.pack(pady=10)
        tk.Label(data_frame, text="Data Directory:", font=("Helvetica", 12), 
                 fg=self.controller.FG_COLOR, bg=self.controller.BG_COLOR).pack(side="left")
        self.data_entry = tk.Entry(data_frame, width=50)
        self.data_entry.insert(0, default_data)
        self.data_entry.pack(side="left", padx=10)
        browse_btn = ttk.Button(data_frame, text="Browse", command=self.browse_data, style="Browse.TButton")
        browse_btn.pack(side="left")

        log_frame = tk.Frame(content, bg=self.controller.BG_COLOR)
        log_frame.pack(pady=10)
        tk.Label(log_frame, text="Log Directory:", font=("Helvetica", 12), 
                 fg=self.controller.FG_COLOR, bg=self.controller.BG_COLOR).pack(side="left")
        self.log_entry = tk.Entry(log_frame, width=50)
        self.log_entry.insert(0, default_logs)
        self.log_entry.pack(side="left", padx=10)
        ttk.Button(log_frame, text="Browse", command=self.browse_logs, style="Browse.TButton").pack(side="left")

        tk.Label(content, text="Log Retention (days):", font=("Helvetica", 12), fg="white", bg=self.controller.BG_COLOR).pack(pady=(15, 0))
        self.log_retention_var = tk.StringVar(value="30")  # default to 30 days
        ttk.Entry(content, textvariable=self.log_retention_var, width=5).pack(pady=(0, 10))

        nav_frame = tk.Frame(content, bg=self.controller.BG_COLOR)
        nav_frame.pack(pady=30)
        ttk.Button(nav_frame, text=self.controller.BACK_BUTTON_TEXT, command=lambda: controller.show_frame(WelcomePage), style="Custom.TButton").pack(side="left", padx=10)
        ttk.Button(nav_frame, text=self.controller.NEXT_BUTTON_TEXT, command=lambda: controller.show_frame(DBCredentialsPage), style="Custom.TButton").pack(side="left", padx=10)

    def browse_data(self):
        directory = filedialog.askdirectory(title="Select Data Directory")
        if directory:
            self.data_entry.delete(0, tk.END)
            self.data_entry.insert(0, directory)

    def browse_logs(self):
        directory = filedialog.askdirectory(title="Select Log Directory")
        if directory:
            self.log_entry.delete(0, tk.END)
            self.log_entry.insert(0, directory)

class DBCredentialsPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller

        title = tk.Label(self, text="🔑 Database Credentials 🔑", font=("Helvetica", 18, "bold"), fg=self.controller.FG_COLOR, bg=self.controller.BG_COLOR)
        title.pack(pady=20)

        form_frame = tk.Frame(self, bg=self.controller.BG_COLOR)
        form_frame.pack(pady=20)

        ttk.Label(form_frame, text="Username:", font=("Helvetica", 12), foreground=self.controller.FG_COLOR, background=self.controller.BG_COLOR).grid(row=0, column=0, sticky="e", padx=10, pady=5)
        self.db_username_var = tk.StringVar(value="admin")
        ttk.Entry(form_frame, textvariable=self.db_username_var, width=40).grid(row=0, column=1, pady=5)

        ttk.Label(form_frame, text="Password:", font=("Helvetica", 12), foreground=self.controller.FG_COLOR, background=self.controller.BG_COLOR).grid(row=1, column=0, sticky="e", padx=10, pady=5)
        self.db_password_var = tk.StringVar(value="secret")
        self.password_entry = ttk.Entry(form_frame, textvariable=self.db_password_var, width=40, show="*")
        self.password_entry.grid(row=1, column=1, pady=5)

        ttk.Label(form_frame, text="Authentication Database:", font=("Helvetica", 12), foreground=self.controller.FG_COLOR, background=self.controller.BG_COLOR).grid(row=2, column=0, sticky="e", padx=10, pady=5)
        self.db_auth_source_var = tk.StringVar(value="admin")
        ttk.Entry(form_frame, textvariable=self.db_auth_source_var, width=40).grid(row=2, column=1, pady=5)

        self.show_pass_var = tk.BooleanVar()
        ttk.Checkbutton(
            form_frame,
            text="Show Password",
            variable=self.show_pass_var,
            command=self.toggle_password,
            style="Custom.TCheckbutton"
        ).grid(row=3, column=1, sticky="w", pady=5)

        # Set custom style for Checkbutton background
        style = ttk.Style()
        style.configure(
            "Custom.TCheckbutton",
            background=self.controller.BG_COLOR,
            foreground=self.controller.FG_COLOR
        )

        nav_frame = tk.Frame(self, bg=self.controller.BG_COLOR)
        nav_frame.pack(pady=30)
        ttk.Button(nav_frame, text=self.controller.BACK_BUTTON_TEXT, command=lambda: controller.show_frame(InstallDirPage), style="Custom.TButton").pack(side="left", padx=10)
        ttk.Button(nav_frame, text=self.controller.NEXT_BUTTON_TEXT, command=lambda: controller.show_frame(DBConnectionPage), style="Custom.TButton").pack(side="left", padx=10)

    def toggle_password(self):
        self.password_entry.config(show="" if self.show_pass_var.get() else "*")

class DBConnectionPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller

        title = tk.Label(self, text="💻 Database Connection Info 💻", font=("Helvetica", 18, "bold"), fg=self.controller.FG_COLOR, bg=self.controller.BG_COLOR)
        title.pack(pady=20)

        form_frame = tk.Frame(self, bg=self.controller.BG_COLOR)
        form_frame.pack(pady=20)

        ttk.Label(form_frame, text="Database Host:", font=("Helvetica", 12), foreground=self.controller.FG_COLOR, background=self.controller.BG_COLOR).grid(row=0, column=0, sticky="e", padx=10, pady=5)
        self.db_host_var = tk.StringVar(value="localhost")
        ttk.Entry(form_frame, textvariable=self.db_host_var, width=40).grid(row=0, column=1, pady=5)

        ttk.Label(form_frame, text="Database Port:", font=("Helvetica", 12), foreground=self.controller.FG_COLOR, background=self.controller.BG_COLOR).grid(row=1, column=0, sticky="e", padx=10, pady=5)
        self.db_port_var = tk.StringVar(value="27017")
        ttk.Entry(form_frame, textvariable=self.db_port_var, width=40).grid(row=1, column=1, pady=5)

        ttk.Button(form_frame, text="🔗 Test DB Connection", command=self.test_database_connection, width=20, style="Custom.TButton").grid(row=2, column=0, columnspan=2, pady=15)

        nav_frame = tk.Frame(self, bg=self.controller.BG_COLOR)
        nav_frame.pack(pady=30)
        ttk.Button(nav_frame, text=self.controller.BACK_BUTTON_TEXT, command=lambda: controller.show_frame(DBCredentialsPage), style="Custom.TButton").pack(side="left", padx=10)
        ttk.Button(nav_frame, text=self.controller.NEXT_BUTTON_TEXT, command=lambda: controller.show_frame(FeaturesPage), style="Custom.TButton").pack(side="left", padx=10)

    def test_database_connection(self):
        host = self.db_host_var.get()
        port = self.db_port_var.get()
        user = self.controller.frames[DBCredentialsPage].db_username_var.get()
        password = self.controller.frames[DBCredentialsPage].db_password_var.get()
        auth_db = self.controller.frames[DBCredentialsPage].db_auth_source_var.get()
        try:
            uri = f"mongodb://{host}:{port}/"
            if user and password:
                uri = f"mongodb://{user}:{password}@{host}:{port}/{auth_db}"
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
            client.server_info()
            messagebox.showinfo("Success", "Connected to MongoDB successfully!")
        except Exception as e:
            messagebox.showerror("Connection Failed", f"Could not connect to MongoDB:\n{e}")

class FeaturesPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller

        # Page title
        title = tk.Label(self, text="⚙️ Select Features ⚙️", font=("Helvetica", 18, "bold"),
                         fg=self.controller.FG_COLOR, bg=self.controller.BG_COLOR)
        title.pack(pady=20)

        # Instructions
        instr = tk.Label(self, text="Select the features you want to enable:",
                         font=("Helvetica", 12), fg=self.controller.FG_COLOR, bg=self.controller.BG_COLOR)
        instr.pack(pady=10)

        # Checkbox variables
        self.data_gathering_var = tk.BooleanVar(value=True)
        self.data_processing_var = tk.BooleanVar(value=True)
        self.ml_training_var = tk.BooleanVar(value=True)

        # Checkboxes
        cb_frame = tk.Frame(self, bg=controller.BG_COLOR)
        cb_frame.pack(pady=20)

        tk.Checkbutton(cb_frame, text="Data Gathering (gather raw data from MAST)",
                       variable=self.data_gathering_var, font=("Helvetica", 12),
                       fg="white", bg=controller.BG_COLOR, selectcolor=controller.BG_COLOR).pack(anchor="w", pady=5)

        tk.Checkbutton(cb_frame, text="Data Processing (prepare light curves for analysis)",
                       variable=self.data_processing_var, font=("Helvetica", 12),
                       fg=self.controller.FG_COLOR, bg=self.controller.BG_COLOR, selectcolor=self.controller.BG_COLOR).pack(anchor="w", pady=5)

        tk.Checkbutton(cb_frame, text="ML Training (train ML algorithm)",
                       variable=self.ml_training_var, font=("Helvetica", 12),
                       fg=self.controller.FG_COLOR, bg=self.controller.BG_COLOR, selectcolor=self.controller.BG_COLOR).pack(anchor="w", pady=5)

        # Navigation buttons
        nav_frame = tk.Frame(self, bg=self.controller.BG_COLOR)
        nav_frame.pack(pady=30)
        ttk.Button(nav_frame, text=self.controller.BACK_BUTTON_TEXT, command=lambda: self.controller.show_frame(DBConnectionPage), style="Custom.TButton").pack(side="left", padx=10)
        ttk.Button(nav_frame, text=self.controller.NEXT_BUTTON_TEXT, command=lambda: self.controller.show_frame(SummaryPage), style="Custom.TButton").pack(side="left", padx=10)

    def go_next(self):
        """Save the selections to controller and proceed to InstallationPage"""
        self.controller.features = {
            "data_gathering": self.data_gathering_var.get(),
            "data_processing": self.data_processing_var.get(),
            "ml_training": self.ml_training_var.get()
        }
        self.controller.show_frame(InstallationPage)

class SummaryPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller

        title = tk.Label(self, text="📋 Summary of Settings 📋", font=("Helvetica", 18, "bold"), fg=self.controller.FG_COLOR, bg=self.controller.BG_COLOR)
        title.pack(pady=20)

        self.summary_text = tk.Text(self, width=90, height=20, font=("Helvetica", 12))
        self.summary_text.pack(padx=20, pady=20)
        self.summary_text.config(state="disabled")

        nav_frame = tk.Frame(self, bg=self.controller.BG_COLOR)
        nav_frame.pack(pady=10)
        ttk.Button(nav_frame, text=self.controller.BACK_BUTTON_TEXT, command=lambda: controller.show_frame(FeaturesPage), style="Custom.TButton").pack(side="left", padx=10)
        ttk.Button(nav_frame, text="Begin Installation", command=lambda: controller.show_frame(InstallationPage), style="Custom.TButton").pack(side="left", padx=10)

    def update_summary(self):
        data_dir = self.controller.frames[InstallDirPage].data_entry.get()
        log_dir = self.controller.frames[InstallDirPage].log_entry.get()
        db_username = self.controller.frames[DBCredentialsPage].db_username_var.get()
        db_password = self.controller.frames[DBCredentialsPage].db_password_var.get()
        db_auth = self.controller.frames[DBCredentialsPage].db_auth_source_var.get()
        db_host = self.controller.frames[DBConnectionPage].db_host_var.get()
        db_port = self.controller.frames[DBConnectionPage].db_port_var.get()
        data_gathering_enabled = self.controller.frames[FeaturesPage].data_gathering_var.get()
        data_processing_enabled = self.controller.frames[FeaturesPage].data_processing_var.get()
        ml_training_enabled = self.controller.frames[FeaturesPage].ml_training_var.get()

        summary = (
            f"Data Directory: {data_dir}\n"
            f"Log Directory: {log_dir}\n"
            f"      Log Retention: {self.controller.frames[InstallDirPage].log_retention_var.get()} days\n\n"
            f"Database Credentials:\n"
            f"      Username: {db_username}\n"
            f"      Password: {'*' * len(db_password)}\n"
            f"      Auth DB: {db_auth}\n\n"
            f"Database Connection:\n"
            f"      Host: {db_host}\n"
            f"      Port: {db_port}\n\n"
            f"Features:\n"
            f"      Data Gathering: {'Enabled' if data_gathering_enabled else 'Disabled'}\n"
            f"      Data Processing: {'Enabled' if data_processing_enabled else 'Disabled'}\n"
            f"      ML Training: {'Enabled' if ml_training_enabled else 'Disabled'}\n"
        )

        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, summary)
        self.summary_text.config(state="disabled")


class InstallationPage(tk.Frame):

    PYTHON_VERSION = "3.12.1"

    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller

        title = tk.Label(self, text="⏳ Installing... ⏳", font=("Helvetica", 18, "bold"), fg=self.controller.FG_COLOR, bg=self.controller.BG_COLOR)
        title.pack(pady=20)

        # Log window
        log_frame = tk.Frame(self, bg=controller.BG_COLOR)
        log_frame.pack(pady=10, fill="both", expand=True)

        self.log_text = tk.Text(log_frame, height=10, width=80,
                                bg="black", fg="lime", insertbackground="white",
                                font=("Consolas", 10), wrap="word", state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

        # Progress bar
        self.progress = ttk.Progressbar(self, orient="horizontal", length=600, mode="determinate")
        self.progress.pack(pady=50)
        style = ttk.Style()
        style.configure("green.Horizontal.TProgressbar", foreground='green', background='green')
        self.progress.config(style="green.Horizontal.TProgressbar")
        self.progress["maximum"] = 12
        self.progress["value"] = 0
        self.progress_value = 0

    def start_installation(self):
        """Start the installation in a separate thread to avoid freezing the GUI."""
        threading.Thread(target=self._run_installation, daemon=True).start()

    def _run_installation(self):
        """Actual installation logic including saving config and updating progress."""
        # Gather all configuration settings
        config = {
            "data_directory": self.controller.frames[InstallDirPage].data_entry.get(),
            "log_directory": self.controller.frames[InstallDirPage].log_entry.get(),
            "log_retention_days": int(self.controller.frames[InstallDirPage].log_retention_var.get()),
            "db_username": self.controller.frames[DBCredentialsPage].db_username_var.get(),
            "db_password": self.controller.frames[DBCredentialsPage].db_password_var.get(),
            "db_auth_database": self.controller.frames[DBCredentialsPage].db_auth_source_var.get(),
            "db_host": self.controller.frames[DBConnectionPage].db_host_var.get(),
            "db_port": self.controller.frames[DBConnectionPage].db_port_var.get(),
            "data_gathering": self.controller.frames[FeaturesPage].data_gathering_var.get(),
            "data_processing": self.controller.frames[FeaturesPage].data_processing_var.get(),
            "ml_training": self.controller.frames[FeaturesPage].ml_training_var.get()
        }

        # Directories
        data_dir = config["data_directory"]
        config_path = os.path.join(data_dir, "config")
        embed_dir = resource_path("python-embed")
        nssm_exe = resource_path("nssm/nssm.exe")
        worker_script = resource_path("worker.py")

        # Ensure embedded Python directory exists
        self.log_message("Ensuring python-embed directory exists...")
        os.makedirs(embed_dir, exist_ok=True)
        self.increment_progress()

        # Download + extract embedded Python
        self.log_message("Downloading and extracting embedded Python...")
        self.download_and_extract_python(embed_dir)
        self.increment_progress()

        python_embed_path = os.path.join(embed_dir, "python.exe")

        # Create configuration file
        try:
            self.log_message("Creating configuration file...")
            os.makedirs(config_path, exist_ok=True)
            config_file_path = os.path.join(config_path, "config.json")
            with open(config_file_path, "w") as f:
                json.dump(config, f, indent=4)
            self.log_message(f"✅ Config file saved at: {config_file_path}")
            self.increment_progress()
        except Exception as e:
            self.log_message(f"❌ Failed to save config file: {e}")
            return

        # Linux systemd service
        if platform.system() == "Linux":
            try:
                self.log_message("Detected Linux OS. Creating systemd service...")
                service_content = f"""[Unit]
Description=Exoplanet Hunter Worker
After=network.target

[Service]
ExecStart={python_embed_path} {worker_script}
WorkingDirectory=/opt/exoplanet_hunter
Restart=always
User=exohunter
Group=exohunter

[Install]
WantedBy=multi-user.target
"""
                service_path = "/etc/systemd/system/exoplanet_hunter.service"
                with open(service_path, "w") as f:
                    f.write(service_content)
                self.increment_progress()

                if self.run_command(["systemctl", "enable", service_path],
                                    success_msg="✅ Successfully enabled systemd service",
                                    error_msg="❌ Failed to enable systemd service"):
                    self.increment_progress()
                if self.run_command(["systemctl", "start", service_path],
                                    success_msg="✅ Successfully started systemd service",
                                    error_msg="❌ Failed to start systemd service"):
                    self.increment_progress()
            except Exception as e:
                self.log_message(f"❌ Linux service setup failed: {e}")
                return

        # Windows service via NSSM
        elif platform.system() == "Windows":
            try:
                self.log_message("Detected Windows OS. Installing service using NSSM...")

                # Ensure NSSM exists
                if not os.path.isfile(nssm_exe):
                    self.log_message(f"❌ NSSM executable not found at {nssm_exe}")
                    return

                # Install NSSM service
                install_cmd = [nssm_exe, "install", "ExoplanetHunterWorker", python_embed_path, worker_script]
                if self.run_command(install_cmd,
                                    success_msg="✅ NSSM Windows service installed successfully!",
                                    error_msg="❌ Failed to install NSSM Windows service"):
                    self.increment_progress()

                # Start NSSM service
                start_cmd = [nssm_exe, "start", "ExoplanetHunterWorker"]
                if self.run_command(start_cmd,
                                    success_msg="✅ NSSM Windows service started successfully!",
                                    error_msg="❌ Failed to start NSSM Windows service"):
                    self.increment_progress()

            except Exception as e:
                self.log_message(f"❌ Windows NSSM service setup failed: {e}")
                return

        # Finish installation
        self.log_message("🎉 Installation complete! Woohoo! 🎉")
        self.controller.show_frame(CompletionPage)

    def log_message(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.update_idletasks()

    def increment_progress(self, amount=1):
        self.progress_value += amount
        if self.progress_value > self.progress["maximum"]:
            self.progress_value = self.progress["maximum"]
        self.progress["value"] = self.progress_value
        self.update_idletasks()

    def run_command(self, command, success_msg=None, error_msg=None):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            if success_msg:
                self.log_message(success_msg)
            if result.stdout.strip():
                self.log_message(f"    Output: {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            self.log_message(f"❌ {error_msg or 'Command failed'}: {e}")
            if e.stdout:
                self.log_message(f"    Stdout: {e.stdout.strip()}")
            if e.stderr:
                self.log_message(f"    Stderr: {e.stderr.strip()}")
            return False
        except FileNotFoundError:
            self.log_message(f"❌ Command not found: {command[0]}")
            return False

    def download_and_extract_python(self, target_dir):
        """Download and extract Python for current OS into target_dir."""
        system = platform.system()
        arch = "amd64" if platform.machine().endswith("64") else "win32"
        os.makedirs(target_dir, exist_ok=True)

        if system == "Windows":
            url = f"https://www.python.org/ftp/python/{self.PYTHON_VERSION}/python-{self.PYTHON_VERSION}-embed-{arch}.zip"
            archive_path = os.path.join(target_dir, "python_embed.zip")
            self.log_message(f"Downloading Windows embedded Python {self.PYTHON_VERSION}...")
            urllib.request.urlretrieve(url, archive_path)
            self.log_message("Extracting embedded Python...")
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            os.remove(archive_path)
            self.log_message(f"Python extracted to {target_dir}")

        elif system == "Linux":
            url = f"https://www.python.org/ftp/python/{self.PYTHON_VERSION}/Python-{self.PYTHON_VERSION}.tgz"
            archive_path = os.path.join(target_dir, f"Python-{self.PYTHON_VERSION}.tgz")
            self.log_message(f"Downloading Linux Python {self.PYTHON_VERSION} source tarball...")
            urllib.request.urlretrieve(url, archive_path)
            self.log_message("Extracting Python source tarball...")
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(target_dir)
            os.remove(archive_path)
            self.log_message(f"Python source extracted to {target_dir}")
            self.log_message("NOTE: On Linux, you may need to compile Python from source.")

        else:
            raise RuntimeError(f"Unsupported OS: {system}")

    def show_frame(self, page_class):
        frame = self.frames[page_class]
        frame.tkraise()
        if page_class == SummaryPage:
            frame.update_summary()
        if page_class == InstallationPage:
            frame.start_installation()


class CompletionPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller

        label = tk.Label(self, text="🎉 Congratulations! 🎉\nYou are now helping find Exoplanets!", 
                         font=("Helvetica", 18, "bold"), fg=self.controller.FG_COLOR, bg=self.controller.BG_COLOR, justify="center")
        label.pack(expand=True)

        close_button = ttk.Button(
            self,
            text="Close Window",
            command=controller.destroy,
            style="Custom.TButton"
        )
        close_button.pack(pady=20)

if __name__ == "__main__":
    app = ExoplanetHunterApp()
    app.mainloop()