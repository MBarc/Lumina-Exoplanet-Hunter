import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
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

