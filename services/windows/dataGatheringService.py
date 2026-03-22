import sys
import os

# Ensure the directory containing this file is on sys.path so that
# dataGatheringServiceLogic and the bundled ml/ package are importable
# regardless of the working directory when the service starts.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import win32serviceutil
import win32service
import win32event
import servicemanager
import threading

from dataGatheringServiceLogic import run_data_gathering


class DataGatheringService(win32serviceutil.ServiceFramework):
    _svc_name_ = "ExoplanetDataGathering"
    _svc_display_name_ = "Exoplanet Data Gathering Service"
    _svc_description_ = "Runs the ExoplanetHunter data_gathering script as a Windows Service."

    def __init__(self, args):
        super().__init__(args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.stop_event = threading.Event()
        self.worker_thread = None

    def SvcStop(self):
        servicemanager.LogInfoMsg("Exoplanet Data Gathering Service stopping...")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.stop_event.set()  # signal the data_gathering loop to stop
        win32event.SetEvent(self.hWaitStop)
        if self.worker_thread:
            self.worker_thread.join()  # wait for graceful shutdown

    def SvcDoRun(self):
        servicemanager.LogInfoMsg("Exoplanet Data Gathering Service starting...")

        # Start the dashboard in a daemon thread (dies automatically when the service stops)
        try:
            import json
            config_path = os.path.join(_THIS_DIR, "..", "config", "config.json")
            config_path = os.path.normpath(config_path)
            with open(config_path) as f:
                cfg = json.load(f)
            from dashboard.app import run as run_dashboard
            dashboard_thread = threading.Thread(
                target=run_dashboard, args=(cfg,), daemon=True, name="LuminaDashboard"
            )
            dashboard_thread.start()
            servicemanager.LogInfoMsg("Lumina dashboard started at http://127.0.0.1:8050")
        except Exception as e:
            servicemanager.LogWarningMsg(f"Dashboard failed to start: {e}")

        self.worker_thread = threading.Thread(target=run_data_gathering, args=(self.stop_event,))
        self.worker_thread.start()

        # Wait until stop event is triggered
        win32event.WaitForSingleObject(self.hWaitStop, win32event.INFINITE)
        servicemanager.LogInfoMsg("Exoplanet Data Gathering Service stopped.")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2 and sys.argv[1] == "run":
        # Debug mode - run like a normal script
        import threading, time
        stop_event = threading.Event()
        try:
            run_data_gathering(stop_event)
        except KeyboardInterrupt:
            print("Keyboard interrupt received, shutting down...")
            stop_event.set()
    else:
        # Normal service handler (install/start/stop/remove)
        win32serviceutil.HandleCommandLine(DataGatheringService)

