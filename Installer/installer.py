import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import subprocess
import threading
import pymongo
import sys


def resource_path(relative_path: str) -> str:
    """Get absolute path to resource (works in dev and PyInstaller)"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)


class LuminaApp(tk.Tk):

    FG_COLOR = "white"
    BG_COLOR = "#0b1a33"
    ACCENT_COLOR = "#00c8ff"
    BACK_BUTTON_TEXT = "Back"
    NEXT_BUTTON_TEXT = "Next"

    def __init__(self):
        super().__init__()
        self.title("Lumina v1.0.0")
        self.geometry("900x600")
        self.resizable(False, False)
        self.container = tk.Frame(self, bg=self.BG_COLOR)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

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
        style.map("Custom.TButton", background=[("active", "#2a4d9c")])

        style.configure(
            "Browse.TButton",
            background=self.BG_COLOR,
            foreground="black",
            padding=(10, 2),
            borderwidth=0,
            focusthickness=3,
            focuscolor="none"
        )
        style.map("Browse.TButton", background=[("active", "#2a4d9c")])

        style.configure(
            "Custom.TCheckbutton",
            background=self.BG_COLOR,
            foreground=self.FG_COLOR
        )

        self.frames = {}
        for F in (WelcomePage, InstallDirPage, DBCredentialsPage, DBConnectionPage,
                  FeaturesPage, SummaryPage, InstallationPage, CompletionPage):
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

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(2, weight=1)

        content = tk.Frame(self, bg=controller.BG_COLOR)
        content.grid(row=1, column=1, sticky="nsew")

        tk.Label(
            content,
            text="Welcome to Lumina",
            font=("Helvetica", 22, "bold"),
            fg=controller.FG_COLOR,
            bg=controller.BG_COLOR
        ).pack(pady=20)

        tk.Label(
            content,
            text=(
                "Lumina turns your computer into an exoplanet search node.\n\n"
                "Once installed, it runs quietly in the background — analyzing stellar\n"
                "light curve data from NASA missions and contributing to the search\n"
                "for worlds beyond our solar system.\n\n"
                "This installer will configure your machine and register the Lumina\n"
                "background service. The process takes about a minute.\n\n"
                "Ready to join the hunt?"
            ),
            justify="center",
            wraplength=650,
            font=("Helvetica", 12),
            fg=controller.FG_COLOR,
            bg=controller.BG_COLOR
        ).pack(padx=20, pady=20)

        ttk.Button(
            content,
            text="Get Started",
            command=lambda: controller.show_frame(InstallDirPage),
            style="Custom.TButton"
        ).pack(pady=20)


class InstallDirPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        content = tk.Frame(self, bg=controller.BG_COLOR)
        content.grid(row=1, column=0)

        tk.Label(
            content, text="Installation Directories",
            font=("Helvetica", 18, "bold"),
            fg=controller.FG_COLOR, bg=controller.BG_COLOR
        ).pack(pady=20)

        data_frame = tk.Frame(content, bg=controller.BG_COLOR)
        data_frame.pack(pady=10)
        tk.Label(data_frame, text="Data Directory:", font=("Helvetica", 12),
                 fg=controller.FG_COLOR, bg=controller.BG_COLOR).pack(side="left")
        self.data_entry = tk.Entry(data_frame, width=50)
        self.data_entry.insert(0, r"C:\Program Files\Lumina\Data")
        self.data_entry.pack(side="left", padx=10)
        ttk.Button(data_frame, text="Browse", command=self.browse_data,
                   style="Browse.TButton").pack(side="left")

        log_frame = tk.Frame(content, bg=controller.BG_COLOR)
        log_frame.pack(pady=10)
        tk.Label(log_frame, text="Log Directory:", font=("Helvetica", 12),
                 fg=controller.FG_COLOR, bg=controller.BG_COLOR).pack(side="left")
        self.log_entry = tk.Entry(log_frame, width=50)
        self.log_entry.insert(0, r"C:\Program Files\Lumina\Logs")
        self.log_entry.pack(side="left", padx=10)
        ttk.Button(log_frame, text="Browse", command=self.browse_logs,
                   style="Browse.TButton").pack(side="left")

        tk.Label(content, text="Log Retention (days):", font=("Helvetica", 12),
                 fg=controller.FG_COLOR, bg=controller.BG_COLOR).pack(pady=(15, 0))
        self.log_retention_var = tk.StringVar(value="30")
        ttk.Entry(content, textvariable=self.log_retention_var, width=5).pack(pady=(0, 10))

        nav_frame = tk.Frame(content, bg=controller.BG_COLOR)
        nav_frame.pack(pady=30)
        ttk.Button(nav_frame, text=controller.BACK_BUTTON_TEXT,
                   command=lambda: controller.show_frame(WelcomePage),
                   style="Custom.TButton").pack(side="left", padx=10)
        ttk.Button(nav_frame, text=controller.NEXT_BUTTON_TEXT,
                   command=self._next, style="Custom.TButton").pack(side="left", padx=10)

    def _next(self):
        data_dir = self.data_entry.get().strip()
        log_dir = self.log_entry.get().strip()
        retention = self.log_retention_var.get().strip()

        if not data_dir:
            messagebox.showerror("Validation Error", "Data directory cannot be empty.")
            return
        if not log_dir:
            messagebox.showerror("Validation Error", "Log directory cannot be empty.")
            return
        if not retention.isdigit() or int(retention) < 1:
            messagebox.showerror("Validation Error", "Log retention must be a positive number of days.")
            return

        self.controller.show_frame(DBCredentialsPage)

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

        tk.Label(self, text="Database Credentials",
                 font=("Helvetica", 18, "bold"),
                 fg=controller.FG_COLOR, bg=controller.BG_COLOR).pack(pady=20)

        form_frame = tk.Frame(self, bg=controller.BG_COLOR)
        form_frame.pack(pady=20)

        ttk.Label(form_frame, text="Username:", font=("Helvetica", 12),
                  foreground=controller.FG_COLOR,
                  background=controller.BG_COLOR).grid(row=0, column=0, sticky="e", padx=10, pady=5)
        self.db_username_var = tk.StringVar(value="admin")
        ttk.Entry(form_frame, textvariable=self.db_username_var,
                  width=40).grid(row=0, column=1, pady=5)

        ttk.Label(form_frame, text="Password:", font=("Helvetica", 12),
                  foreground=controller.FG_COLOR,
                  background=controller.BG_COLOR).grid(row=1, column=0, sticky="e", padx=10, pady=5)
        self.db_password_var = tk.StringVar()
        self.password_entry = ttk.Entry(form_frame, textvariable=self.db_password_var,
                                        width=40, show="*")
        self.password_entry.grid(row=1, column=1, pady=5)

        ttk.Label(form_frame, text="Authentication Database:", font=("Helvetica", 12),
                  foreground=controller.FG_COLOR,
                  background=controller.BG_COLOR).grid(row=2, column=0, sticky="e", padx=10, pady=5)
        self.db_auth_source_var = tk.StringVar(value="admin")
        ttk.Entry(form_frame, textvariable=self.db_auth_source_var,
                  width=40).grid(row=2, column=1, pady=5)

        self.show_pass_var = tk.BooleanVar()
        ttk.Checkbutton(
            form_frame, text="Show Password",
            variable=self.show_pass_var,
            command=self.toggle_password,
            style="Custom.TCheckbutton"
        ).grid(row=3, column=1, sticky="w", pady=5)

        nav_frame = tk.Frame(self, bg=controller.BG_COLOR)
        nav_frame.pack(pady=30)
        ttk.Button(nav_frame, text=controller.BACK_BUTTON_TEXT,
                   command=lambda: controller.show_frame(InstallDirPage),
                   style="Custom.TButton").pack(side="left", padx=10)
        ttk.Button(nav_frame, text=controller.NEXT_BUTTON_TEXT,
                   command=self._next, style="Custom.TButton").pack(side="left", padx=10)

    def _next(self):
        if not self.db_username_var.get().strip():
            messagebox.showerror("Validation Error", "Database username cannot be empty.")
            return
        if not self.db_auth_source_var.get().strip():
            messagebox.showerror("Validation Error", "Authentication database cannot be empty.")
            return
        self.controller.show_frame(DBConnectionPage)

    def toggle_password(self):
        self.password_entry.config(show="" if self.show_pass_var.get() else "*")


class DBConnectionPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller

        tk.Label(self, text="Database Connection",
                 font=("Helvetica", 18, "bold"),
                 fg=controller.FG_COLOR, bg=controller.BG_COLOR).pack(pady=20)

        form_frame = tk.Frame(self, bg=controller.BG_COLOR)
        form_frame.pack(pady=20)

        ttk.Label(form_frame, text="Database Host:", font=("Helvetica", 12),
                  foreground=controller.FG_COLOR,
                  background=controller.BG_COLOR).grid(row=0, column=0, sticky="e", padx=10, pady=5)
        self.db_host_var = tk.StringVar(value="localhost")
        ttk.Entry(form_frame, textvariable=self.db_host_var,
                  width=40).grid(row=0, column=1, pady=5)

        ttk.Label(form_frame, text="Database Port:", font=("Helvetica", 12),
                  foreground=controller.FG_COLOR,
                  background=controller.BG_COLOR).grid(row=1, column=0, sticky="e", padx=10, pady=5)
        self.db_port_var = tk.StringVar(value="27017")
        ttk.Entry(form_frame, textvariable=self.db_port_var,
                  width=40).grid(row=1, column=1, pady=5)

        ttk.Button(form_frame, text="Test Connection",
                   command=self.test_database_connection,
                   width=20, style="Custom.TButton").grid(row=2, column=0, columnspan=2, pady=15)

        nav_frame = tk.Frame(self, bg=controller.BG_COLOR)
        nav_frame.pack(pady=30)
        ttk.Button(nav_frame, text=controller.BACK_BUTTON_TEXT,
                   command=lambda: controller.show_frame(DBCredentialsPage),
                   style="Custom.TButton").pack(side="left", padx=10)
        ttk.Button(nav_frame, text=controller.NEXT_BUTTON_TEXT,
                   command=self._next, style="Custom.TButton").pack(side="left", padx=10)

    def _next(self):
        host = self.db_host_var.get().strip()
        port = self.db_port_var.get().strip()

        if not host:
            messagebox.showerror("Validation Error", "Database host cannot be empty.")
            return
        if not port.isdigit() or not (1 <= int(port) <= 65535):
            messagebox.showerror("Validation Error", "Port must be a number between 1 and 65535.")
            return

        self.controller.show_frame(FeaturesPage)

    def test_database_connection(self):
        host = self.db_host_var.get().strip()
        port = self.db_port_var.get().strip()

        if not host or not port.isdigit():
            messagebox.showerror("Validation Error", "Please enter a valid host and port before testing.")
            return

        user = self.controller.frames[DBCredentialsPage].db_username_var.get()
        password = self.controller.frames[DBCredentialsPage].db_password_var.get()
        auth_db = self.controller.frames[DBCredentialsPage].db_auth_source_var.get()

        try:
            if user and password:
                uri = f"mongodb://{user}:{password}@{host}:{port}/{auth_db}"
            else:
                uri = f"mongodb://{host}:{port}/"
            client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
            client.server_info()
            messagebox.showinfo("Success", "Connected to MongoDB successfully!")
        except Exception as e:
            messagebox.showerror("Connection Failed", f"Could not connect to MongoDB:\n{e}")


class FeaturesPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller

        tk.Label(self, text="Select Features",
                 font=("Helvetica", 18, "bold"),
                 fg=controller.FG_COLOR, bg=controller.BG_COLOR).pack(pady=20)

        tk.Label(self, text="Choose which services this node will run:",
                 font=("Helvetica", 12),
                 fg=controller.FG_COLOR, bg=controller.BG_COLOR).pack(pady=10)

        cb_frame = tk.Frame(self, bg=controller.BG_COLOR)
        cb_frame.pack(pady=20)

        self.data_gathering_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            cb_frame,
            text="Data Gathering  —  retrieve light curve data from mission archives",
            variable=self.data_gathering_var,
            font=("Helvetica", 12),
            fg=controller.FG_COLOR, bg=controller.BG_COLOR,
            selectcolor=controller.BG_COLOR
        ).pack(anchor="w", pady=5)

        self.data_processing_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            cb_frame,
            text="Data Processing  —  prepare and screen light curves for transit signals",
            variable=self.data_processing_var,
            font=("Helvetica", 12),
            fg=controller.FG_COLOR, bg=controller.BG_COLOR,
            selectcolor=controller.BG_COLOR
        ).pack(anchor="w", pady=5)

        nav_frame = tk.Frame(self, bg=controller.BG_COLOR)
        nav_frame.pack(pady=30)
        ttk.Button(nav_frame, text=controller.BACK_BUTTON_TEXT,
                   command=lambda: controller.show_frame(DBConnectionPage),
                   style="Custom.TButton").pack(side="left", padx=10)
        ttk.Button(nav_frame, text=controller.NEXT_BUTTON_TEXT,
                   command=self._next, style="Custom.TButton").pack(side="left", padx=10)

    def _next(self):
        if not self.data_gathering_var.get() and not self.data_processing_var.get():
            messagebox.showerror("Validation Error", "At least one feature must be selected.")
            return
        self.controller.show_frame(SummaryPage)


class SummaryPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller

        tk.Label(self, text="Review Settings",
                 font=("Helvetica", 18, "bold"),
                 fg=controller.FG_COLOR, bg=controller.BG_COLOR).pack(pady=20)

        self.summary_text = tk.Text(self, width=90, height=18,
                                    font=("Consolas", 11),
                                    bg="#060d1f", fg="white",
                                    relief="flat", state="disabled")
        self.summary_text.pack(padx=20, pady=10)

        nav_frame = tk.Frame(self, bg=controller.BG_COLOR)
        nav_frame.pack(pady=10)
        ttk.Button(nav_frame, text=controller.BACK_BUTTON_TEXT,
                   command=lambda: controller.show_frame(FeaturesPage),
                   style="Custom.TButton").pack(side="left", padx=10)
        ttk.Button(nav_frame, text="Begin Installation",
                   command=lambda: controller.show_frame(InstallationPage),
                   style="Custom.TButton").pack(side="left", padx=10)

    def update_summary(self):
        data_dir = self.controller.frames[InstallDirPage].data_entry.get()
        log_dir = self.controller.frames[InstallDirPage].log_entry.get()
        retention = self.controller.frames[InstallDirPage].log_retention_var.get()
        db_username = self.controller.frames[DBCredentialsPage].db_username_var.get()
        db_password = self.controller.frames[DBCredentialsPage].db_password_var.get()
        db_auth = self.controller.frames[DBCredentialsPage].db_auth_source_var.get()
        db_host = self.controller.frames[DBConnectionPage].db_host_var.get()
        db_port = self.controller.frames[DBConnectionPage].db_port_var.get()
        gathering = self.controller.frames[FeaturesPage].data_gathering_var.get()
        processing = self.controller.frames[FeaturesPage].data_processing_var.get()

        summary = (
            f"Directories\n"
            f"  Data:           {data_dir}\n"
            f"  Logs:           {log_dir}\n"
            f"  Log Retention:  {retention} days\n\n"
            f"Database\n"
            f"  Host:           {db_host}:{db_port}\n"
            f"  Username:       {db_username}\n"
            f"  Password:       {'*' * len(db_password)}\n"
            f"  Auth Database:  {db_auth}\n\n"
            f"Features\n"
            f"  Data Gathering:   {'Enabled' if gathering else 'Disabled'}\n"
            f"  Data Processing:  {'Enabled' if processing else 'Disabled'}\n"
        )

        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, summary)
        self.summary_text.config(state="disabled")


class InstallationPage(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller
        self._started = False

        tk.Label(self, text="Installing...",
                 font=("Helvetica", 18, "bold"),
                 fg=controller.FG_COLOR, bg=controller.BG_COLOR).pack(pady=20)

        log_frame = tk.Frame(self, bg=controller.BG_COLOR)
        log_frame.pack(pady=10, fill="both", expand=True)

        self.log_text = tk.Text(
            log_frame, height=10, width=80,
            bg="black", fg="lime", insertbackground="white",
            font=("Consolas", 10), wrap="word", state="disabled"
        )
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

        self.progress = ttk.Progressbar(self, orient="horizontal", length=600, mode="determinate")
        self.progress.pack(pady=30)
        style = ttk.Style()
        style.configure("green.Horizontal.TProgressbar", foreground="green", background="green")
        self.progress.config(style="green.Horizontal.TProgressbar")
        self.progress["maximum"] = 5
        self.progress["value"] = 0
        self.progress_value = 0

    def start_installation(self):
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._run_installation, daemon=True).start()

    def _run_installation(self):
        frames = self.controller.frames

        config = {
            "data_directory": frames[InstallDirPage].data_entry.get(),
            "log_directory": frames[InstallDirPage].log_entry.get(),
            "log_retention_days": int(frames[InstallDirPage].log_retention_var.get()),
            "db_username": frames[DBCredentialsPage].db_username_var.get(),
            "db_password": frames[DBCredentialsPage].db_password_var.get(),
            "db_auth_database": frames[DBCredentialsPage].db_auth_source_var.get(),
            "db_host": frames[DBConnectionPage].db_host_var.get(),
            "db_port": frames[DBConnectionPage].db_port_var.get(),
            "data_gathering": frames[FeaturesPage].data_gathering_var.get(),
            "data_processing": frames[FeaturesPage].data_processing_var.get(),
        }

        data_dir = config["data_directory"]
        config_dir = os.path.join(data_dir, "config")
        services_dir = os.path.join(data_dir, "services")
        python_exe = os.path.join(resource_path("python-embed"), "python.exe")
        service_src = resource_path(os.path.join("services", "windows", "dataGatheringService.py"))
        service_logic_src = resource_path(os.path.join("services", "windows", "dataGatheringServiceLogic.py"))
        service_dest = os.path.join(services_dir, "dataGatheringService.py")

        # Step 1 — Create directories
        try:
            self.log_message("Creating installation directories...")
            os.makedirs(config_dir, exist_ok=True)
            os.makedirs(services_dir, exist_ok=True)
            self.log_message(f"  {data_dir}")
            self.increment_progress()
        except Exception as e:
            self.log_message(f"Failed to create directories: {e}")
            return

        # Step 2 — Copy service files
        try:
            self.log_message("Copying service files...")
            import shutil
            shutil.copy2(service_src, services_dir)
            shutil.copy2(service_logic_src, services_dir)
            self.log_message(f"  Service files copied to {services_dir}")
            self.increment_progress()
        except Exception as e:
            self.log_message(f"Failed to copy service files: {e}")
            return

        # Step 3 — Write config
        try:
            self.log_message("Writing configuration file...")
            config_path = os.path.join(config_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            self.log_message(f"  Config saved to {config_path}")
            self.increment_progress()
        except Exception as e:
            self.log_message(f"Failed to write config: {e}")
            return

        # Step 4 — Register Windows service
        self.log_message("Registering Lumina background service...")
        if not self.run_command(
            [python_exe, service_dest, "install"],
            success_msg="  Service registered successfully.",
            error_msg="Failed to register service"
        ):
            return
        self.increment_progress()

        # Step 5 — Start service
        self.log_message("Starting Lumina service...")
        if not self.run_command(
            ["net", "start", "ExoplanetDataGathering"],
            success_msg="  Service started.",
            error_msg="Failed to start service"
        ):
            return
        self.increment_progress()

        self.log_message("Installation complete.")
        self.controller.show_frame(CompletionPage)

    def log_message(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.update_idletasks()

    def increment_progress(self, amount=1):
        self.progress_value = min(self.progress_value + amount, self.progress["maximum"])
        self.progress["value"] = self.progress_value
        self.update_idletasks()

    def run_command(self, command, success_msg=None, error_msg=None):
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            if success_msg:
                self.log_message(success_msg)
            if result.stdout.strip():
                self.log_message(f"  {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            self.log_message(f"  {error_msg or 'Command failed'}: {e}")
            if e.stderr:
                self.log_message(f"  {e.stderr.strip()}")
            return False
        except FileNotFoundError:
            self.log_message(f"  Executable not found: {command[0]}")
            return False


class CompletionPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.BG_COLOR)
        self.controller = controller

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        content = tk.Frame(self, bg=controller.BG_COLOR)
        content.grid(row=0, column=0)

        tk.Label(
            content,
            text="Lumina is installed.",
            font=("Helvetica", 22, "bold"),
            fg=controller.FG_COLOR, bg=controller.BG_COLOR
        ).pack(pady=20)

        tk.Label(
            content,
            text=(
                "The background service is running. Your machine is now part of ExoNet\n"
                "and will begin analyzing stellar light curves automatically.\n\n"
                "You can monitor progress from the Lumina dashboard at:\n"
                "http://localhost:8050"
            ),
            font=("Helvetica", 12),
            justify="center",
            fg=controller.FG_COLOR, bg=controller.BG_COLOR
        ).pack(pady=10)

        ttk.Button(
            content,
            text="Close",
            command=controller.destroy,
            style="Custom.TButton"
        ).pack(pady=30)


if __name__ == "__main__":
    app = LuminaApp()
    app.mainloop()
