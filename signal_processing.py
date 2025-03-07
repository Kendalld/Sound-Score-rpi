"""
Digital Signal Processing Module

This module provides real-time digital signal processing capabilities for audio analysis.
Features include:
- Low-pass filtering using Butterworth filter
- FFT computation with windowing
- Frequency spectrum visualization
- Data logging of frequency analysis
- Signal statistics and metrics

The processing is optimized for embedded systems and handles ADC input values (0-1023).
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import os
import csv
from typing import Tuple, List, Optional

class SignalProcessor:
    """
    Real-time digital signal processor for audio analysis.
    
    Provides filtering and frequency analysis capabilities, optimized for
    embedded systems processing ADC data. Includes visualization and data
    logging features.
    
    Attributes:
        sample_rate (int): Audio sampling rate in Hz
        cutoff_freq (int): Low-pass filter cutoff frequency in Hz
        fft_data (list): Storage for FFT analysis results
        filter_order (int): Order of the Butterworth filter
        window_type (str): Type of window function for FFT
    """
    
    def __init__(self, sample_rate: int = 44100, cutoff_freq: int = 5000, 
                 filter_order: int = 4, window_type: str = 'hanning'):
        """
        Initialize signal processor with configurable parameters.
        
        Args:
            sample_rate: Sampling frequency in Hz
            cutoff_freq: Low-pass filter cutoff frequency in Hz
            filter_order: Butterworth filter order (higher = sharper cutoff)
            window_type: FFT window function ('hanning', 'hamming', 'blackman')
        """
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.filter_order = filter_order
        self.window_type = window_type
        self.fft_data = []
        
        # Pre-compute filter coefficients
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = self.cutoff_freq / nyquist
        self.b, self.a = signal.butter(self.filter_order, normal_cutoff, 
                                     btype='low', analog=False)
        
        # Create output directory
        os.makedirs("data", exist_ok=True)

    def get_window(self, size: int) -> np.ndarray:
        """
        Get the specified window function.
        
        Args:
            size: Length of the window
            
        Returns:
            Window function array
        """
        if self.window_type == 'hanning':
            return np.hanning(size)
        elif self.window_type == 'hamming':
            return np.hamming(size)
        elif self.window_type == 'blackman':
            return np.blackman(size)
        else:
            return np.hanning(size)  # Default to Hanning

    def apply_low_pass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Butterworth low-pass filter to input data.
        
        Implements a Butterworth filter to remove high-frequency noise.
        Input values are clipped to valid ADC range (0-1023) after filtering.
        
        Args:
            data: Input signal data
            
        Returns:
            Filtered signal data, clipped to ADC range
        
        Note:
            Uses pre-computed filter coefficients for efficiency
        """
        if len(data) == 0:
            return data

        try:
            # Apply filter (convert to float for processing)
            filtered_data = signal.filtfilt(self.b, self.a, data.astype(float))
            
            # Ensure output remains in valid ADC range
            return np.clip(filtered_data, 0, 1023).astype(int)
            
        except Exception as e:
            print(f"⚠️ Warning: Filter application failed: {str(e)}")
            return data

    def compute_signal_stats(self, data: np.ndarray) -> dict:
        """
        Compute basic signal statistics.
        
        Args:
            data: Input signal data
            
        Returns:
            Dictionary containing signal statistics:
            - mean: Average signal level
            - std: Standard deviation
            - rms: Root mean square value
            - peak: Peak value
            - peak_to_peak: Peak-to-peak range
        """
        if len(data) == 0:
            return {}
            
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'rms': np.sqrt(np.mean(np.square(data))),
            'peak': np.max(np.abs(data)),
            'peak_to_peak': np.ptp(data)
        }

    def compute_fft(self, audio_data: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Compute FFT of audio data with window function.
        
        Applies windowing to reduce spectral leakage and computes the
        frequency spectrum. Results are stored for later visualization
        and analysis.
        
        Args:
            audio_data: Input audio samples
            
        Returns:
            Tuple of (frequencies, magnitudes) if successful, None otherwise
        """
        if len(audio_data) == 0:
            print("⚠️ Warning: No audio data available for FFT.")
            return None

        try:
            # Apply window function
            window = self.get_window(len(audio_data))
            windowed_audio = audio_data * window

            # Compute FFT
            fft_spectrum = np.fft.fft(windowed_audio)
            freq = np.fft.fftfreq(len(fft_spectrum), d=1/self.sample_rate)
            
            # Compute magnitude spectrum (normalize by window RMS)
            magnitude = np.abs(fft_spectrum) / np.sqrt(np.mean(np.square(window)))

            # Store positive frequencies only
            pos_freq = freq[:len(freq)//2]
            pos_magnitude = magnitude[:len(magnitude)//2]
            
            self.fft_data.append((pos_freq, pos_magnitude))
            return pos_freq, pos_magnitude
            
        except Exception as e:
            print(f"⚠️ Warning: FFT computation failed: {str(e)}")
            return None

    def save_fft_plot(self, filename: str = "data/audio_fft.png", 
                     log_scale: bool = True) -> None:
        """
        Generate and save frequency spectrum visualization.
        
        Creates a plot showing the magnitude spectrum of the most recent
        FFT analysis. The plot includes proper axis labels and grid.
        
        Args:
            filename: Output image file path
            log_scale: Whether to use logarithmic scale for magnitude
        """
        if not self.fft_data:
            return

        # Get most recent FFT data
        freq, magnitude = self.fft_data[-1]
        
        # Create frequency spectrum plot
        plt.figure(figsize=(12, 6))
        
        # Main magnitude plot
        plt.subplot(2, 1, 1)
        if log_scale:
            plt.semilogy(freq, magnitude)
        else:
            plt.plot(freq, magnitude)
            
        plt.title("Frequency Spectrum (FFT)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.grid(True)
        
        # Add spectrogram
        plt.subplot(2, 1, 2)
        if len(self.fft_data) > 1:
            magnitudes = np.array([m for _, m in self.fft_data])
            plt.imshow(magnitudes.T, aspect='auto', origin='lower',
                      extent=[0, len(self.fft_data), freq[0], freq[-1]])
            plt.colorbar(label='Magnitude')
            plt.ylabel("Frequency (Hz)")
            plt.xlabel("Time (frames)")
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✅ FFT plot saved to {filename}")

    def save_fft_data(self, filename: str = "data/fft_data.csv") -> None:
        """
        Save frequency spectrum data to CSV file.
        
        Exports the frequency and magnitude data from the most recent
        FFT analysis for further processing or analysis.
        
        Args:
            filename: Output CSV file path
        """
        if not self.fft_data:
            return

        freq, magnitude = self.fft_data[-1]
        stats = self.compute_signal_stats(magnitude)
        
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header with signal statistics
            writer.writerow(["Signal Statistics"])
            for stat, value in stats.items():
                writer.writerow([stat, f"{value:.2f}"])
            
            # Write frequency data
            writer.writerow([])  # Empty row for separation
            writer.writerow(["Frequency (Hz)", "Magnitude"])
            for f, m in zip(freq, magnitude):
                writer.writerow([f"{f:.1f}", f"{m:.6f}"])

        print(f"✅ FFT data saved to {filename}")
