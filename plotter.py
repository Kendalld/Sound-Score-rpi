"""
Sound Level Monitoring and Analysis System

This module provides real-time sound level monitoring, analysis, and visualization.
It includes:
- Noise level scoring based on signal variation
- Real-time signal plotting and data logging
- Spike detection for sudden sound changes
- Average intensity tracking
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from collections import deque
import time

class NoiseScorer:
    """
    Analyzes audio signal variations to produce a normalized noise score.
    
    The scorer maintains a rolling buffer of samples and computes variations
    over time to determine the noise level in an environment. It uses an
    adaptive calibration phase to establish baseline noise levels.
    
    Attributes:
        window_size (int): Number of seconds to keep in variation history
        baseline_floor (float): Minimum variation threshold
        baseline_ceiling (float): Maximum variation threshold (set during calibration)
    """
    
    def __init__(self, window_size=10, floor=5.0, ceiling=None):
        # Window settings
        self.window_size = window_size
        self.sample_buffer = deque(maxlen=1000)  # 1000 most recent samples
        self.variation_history = deque(maxlen=window_size)
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_count = 0
        
        # Thresholds
        self.baseline_floor = floor
        self.baseline_ceiling = ceiling
        
    def calibrate(self, value):
        """
        Collect samples and establish baseline variation levels.
        
        Args:
            value (float): New ADC sample value to use for calibration
        """
        self.calibration_count += 1
        if len(self.sample_buffer) < 1000:
            self.sample_buffer.append(value)
            if self.calibration_count % 100 == 0:
                print(f"\rCalibrating: {len(self.sample_buffer)}/1000 samples", end='', flush=True)
        elif not self.is_calibrated:
            signal = np.array(self.sample_buffer)
            current_variation = np.std(signal)
            
            if self.baseline_ceiling is None:
                self.baseline_ceiling = max(current_variation * 4, 20.0)
            
            self.is_calibrated = True
            print(f"\n✅ Noise calibration complete:")
            print(f"   Floor variation: {self.baseline_floor:.1f}")
            print(f"   Ceiling variation: {self.baseline_ceiling:.1f}")
            print(f"   Current variation: {current_variation:.1f}")
            print(f"   Mean signal level: {np.mean(signal):.1f}\n")
    
    def add_sample(self, value):
        """
        Add a new sample and update variation history.
        
        Args:
            value (float): New ADC sample value
        """
        self.sample_buffer.append(value)
        
        if len(self.sample_buffer) >= 100:
            recent_signal = np.array(list(self.sample_buffer)[-100:])
            variation = np.std(recent_signal)
            self.variation_history.append(variation)
    
    def get_score(self):
        """
        Calculate noise score from 1-100 based on current variation.
        
        Returns:
            int: Noise score between 1 and 100
        """
        if not self.variation_history or not self.is_calibrated:
            return 0
            
        current_variation = np.mean(self.variation_history)
        normalized = (current_variation - self.baseline_floor) / (self.baseline_ceiling - self.baseline_floor)
        score = int(normalized * 100)
        return max(1, min(100, score))

class LivePlot:
    """
    Real-time signal plotting and analysis with data logging.
    
    Provides live visualization of audio signals, noise scoring, and spike detection.
    Data is logged to CSV files for later analysis.
    
    Attributes:
        buffer_size (int): Number of samples to keep in plotting buffer
        score_update_interval (float): Seconds between score updates
        spike_threshold (float): Minimum change to count as a spike
        min_spike_interval (float): Minimum seconds between spikes
    """
    
    def __init__(self, buffer_size=500, save_file="plot_data.csv", 
                 image_file="plot.png", score_update_interval=0.5):
        # Buffer settings
        self.buffer_size = buffer_size
        self.data = deque([0] * buffer_size, maxlen=buffer_size)
        self.score_update_interval = score_update_interval
        self.last_score_time = time.time()
        
        # Analysis components
        self.noise_scorer = NoiseScorer()
        
        # Intensity tracking
        self.total_samples = 0
        self.running_sum = 0
        self.avg_intensity = 0
        
        # Spike detection settings
        self.spike_threshold = 40
        self.spike_count = 0
        self.last_value = None
        self.last_spike_time = time.time()
        self.min_spike_interval = 0.1
        self.smoothing_window = deque(maxlen=5)
        
        # File setup
        self._setup_files(save_file, image_file)
        
    def _setup_files(self, save_file, image_file):
        """Set up data logging files."""
        os.makedirs("data", exist_ok=True)
        self.csv_path = f"data/{save_file}"
        self.image_path = f"data/{image_file}"
        self.scores_path = f"data/noise_scores.csv"
        
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Time", "ADC Value", "Variation"])
        
        self.scores_file = open(self.scores_path, "w", newline="")
        self.scores_writer = csv.writer(self.scores_file)
        self.scores_writer.writerow(["Time", "Noise Score", "Variation", 
                                   "Avg Intensity", "Spike Count"])

    def get_smoothed_value(self, new_value):
        """Apply rolling average smoothing to new values."""
        self.smoothing_window.append(new_value)
        return np.mean(self.smoothing_window)

    def update_plot(self, new_value, timestamp):
        """
        Update plot data and detect signal changes.
        
        Args:
            new_value (float): New ADC sample value
            timestamp (float): Current timestamp
        """
        self.data.append(new_value)
        
        # Update running statistics
        self.total_samples += 1
        self.running_sum += new_value
        self.avg_intensity = self.running_sum / self.total_samples
        
        # Spike detection
        smoothed_value = self.get_smoothed_value(new_value)
        current_time = time.time()
        
        if self.last_value is not None:
            time_since_last_spike = current_time - self.last_spike_time
            if (time_since_last_spike >= self.min_spike_interval and 
                abs(smoothed_value - self.last_value) > self.spike_threshold):
                self.spike_count += 1
                self.last_spike_time = current_time
        self.last_value = smoothed_value
        
        # Calculate current variation
        if len(self.data) >= 100:
            recent_data = np.array(list(self.data)[-100:])
            variation = np.std(recent_data)
        else:
            variation = 0
            
        self.csv_writer.writerow([timestamp, new_value, variation])

    def update_noise_score(self, new_value, timestamp):
        """
        Update and display current noise score.
        
        Args:
            new_value (float): New ADC sample value
            timestamp (float): Current timestamp
        """
        if not self.noise_scorer.is_calibrated:
            self.noise_scorer.calibrate(new_value)
            return
        
        self.noise_scorer.add_sample(new_value)
        
        current_time = time.time()
        if current_time - self.last_score_time >= self.score_update_interval:
            score = self.noise_scorer.get_score()
            current_variation = (np.mean(self.noise_scorer.variation_history) 
                               if self.noise_scorer.variation_history else 0)
            
            self.scores_writer.writerow([timestamp, score, current_variation, 
                                       self.avg_intensity, self.spike_count])
            
            # Create visual display
            bar_length = int(score / 2)
            bar = '█' * bar_length
            desc = self._get_noise_description(score)
            
            print(f"\rNoise Level: {score:3d}/100 |{bar:<50}| {desc} | "
                  f"Avg: {self.avg_intensity:.1f} | Spikes: {self.spike_count}", 
                  end='', flush=True)
            
            self.last_score_time = current_time

    def _get_noise_description(self, score):
        """Get text description of noise level."""
        if score < 20:
            return "Very Quiet"
        elif score < 40:
            return "Quiet"
        elif score < 60:
            return "Normal"
        elif score < 80:
            return "Loud"
        else:
            return "Very Loud"

    def save_plot(self):
        """Generate and save visualization plots."""
        plt.figure(figsize=(10, 4))
        
        # Signal plot
        plt.subplot(2, 1, 1)
        plt.plot(list(self.data), color='blue')
        plt.title("ADC Signal")
        plt.xlabel("Samples")
        plt.ylabel("ADC Value")
        plt.ylim(0, 100)
        plt.grid()
        
        plt.axhline(y=self.avg_intensity, color='r', linestyle='--', 
                   label=f'Avg: {self.avg_intensity:.1f}')
        plt.legend()
        
        # Change distribution plot
        plt.subplot(2, 1, 2)
        values = np.array(list(self.data))
        diffs = np.abs(np.diff(values))
        plt.hist(diffs, bins=20, color='green', alpha=0.7)
        plt.title(f"Change Distribution (Spikes: {self.spike_count})")
        plt.xlabel("Change Magnitude")
        plt.ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(self.image_path)
        plt.close()
        print(f"Plot saved to {self.image_path}")

    def save_and_close(self):
        """Save final plots and close all files."""
        self.csv_file.close()
        self.scores_file.close()
        self.save_plot()
        print(f"Plot data saved to {self.csv_path}")
        print(f"Noise scores saved to {self.scores_path}")
