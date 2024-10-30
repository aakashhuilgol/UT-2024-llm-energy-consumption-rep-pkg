# power_metrics_tracker.py
import subprocess
import time
from tqdm import tqdm

class PowerMetricsTracker:
    def __init__(self, idle_duration):
        self.powermetrics_process = None
        self.idle_duration = idle_duration # put at 60 for experiments

    def run_powermetrics(self, output):
        """
        Starts powermetrics as a subprocess and saves the output to a file.
        """
        self.powermetrics_process = subprocess.Popen(
            ['sudo', 'powermetrics', '-i', '1000', '--samplers', 'cpu_power,gpu_power', '-a',
             '--hide-cpu-duty-cycle', '--show-usage-summary', '--show-extra-power-info', '-o', output],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Power metrics tracking started.")
        self.track_time(self.idle_duration)

    def stop_powermetrics(self):
        """
        Terminates the powermetrics process.
        """
        self.track_time(self.idle_duration)
        if self.powermetrics_process:
            self.powermetrics_process.terminate()
            print("Power metrics tracking stopped.")

    def track_time(self, duration):
        """
        Tracks progress for the specified duration with tqdm.
        
        Args:
            duration (int): Duration to track in seconds.
        """
        for _ in tqdm(range(duration), desc="Idle State", unit="s"):
            time.sleep(1)