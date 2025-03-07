"""
Audio Recording Module

This module handles the conversion and storage of ADC samples to WAV format.
It is designed to work with 10-bit ADC data (0-1023) and converts it to
16-bit PCM audio format (-32768 to 32767) for standard WAV file compatibility.

Features:
- Real-time ADC to PCM conversion
- Automatic value scaling and clipping
- WAV file creation with proper headers
- Support for various sample rates
- Audio level monitoring
- DC offset removal
- Optional compression
"""

import wave
import os
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime

class AudioWriter:
    """
    Audio sample converter and WAV file writer.
    
    Handles the conversion of ADC samples to PCM format and writes them
    to a WAV file. Includes automatic range conversion and error checking.
    
    Attributes:
        filename (str): Path to output WAV file
        sample_rate (int): Audio sampling rate in Hz
        samples (list): Buffer for collected audio samples
        dc_offset (float): Running DC offset for correction
        peak_value (int): Maximum absolute sample value seen
        compression_factor (float): Dynamic range compression factor
    """
    
    def __init__(self, filename: str = "audio_output.wav", 
                 sample_rate: int = 44100, buffer_size: int = 1000,
                 remove_dc: bool = True, compression: float = 1.0):
        """
        Initialize audio writer with specified parameters.
        
        Args:
            filename: Name of output WAV file (default: audio_output.wav)
            sample_rate: Sampling rate in Hz (default: 44100)
            buffer_size: Size of internal sample buffer
            remove_dc: Whether to remove DC offset (default: True)
            compression: Compression factor (1.0 = no compression)
        """
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"data/audio_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.filename = f"{self.output_dir}/{filename}"
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.remove_dc = remove_dc
        self.compression_factor = max(0.1, min(compression, 10.0))
        
        # Initialize buffers and statistics
        self.samples: List[int] = []
        self.buffer: List[int] = []
        self.dc_offset = 0.0
        self.peak_value = 0
        self.stats: Dict[str, float] = {
            'min': float('inf'),
            'max': float('-inf'),
            'sum': 0,
            'sum_squared': 0,
            'count': 0
        }

    def _update_stats(self, value: float) -> None:
        """
        Update running statistics for audio monitoring.
        
        Args:
            value: New sample value
        """
        self.stats['min'] = min(self.stats['min'], value)
        self.stats['max'] = max(self.stats['max'], value)
        self.stats['sum'] += value
        self.stats['sum_squared'] += value * value
        self.stats['count'] += 1
        
        abs_value = abs(value)
        if abs_value > self.peak_value:
            self.peak_value = abs_value

    def _apply_compression(self, value: float) -> float:
        """
        Apply dynamic range compression.
        
        Args:
            value: Input sample value
            
        Returns:
            Compressed sample value
        """
        if self.compression_factor == 1.0:
            return value
            
        # Compress using power law
        sign = np.sign(value)
        normalized = abs(value) / 32768.0
        compressed = sign * 32768.0 * (normalized ** self.compression_factor)
        return compressed

    def add_sample(self, adc_value: int) -> None:
        """
        Convert and store a single ADC sample.
        
        Converts a 10-bit ADC value (0-1023) to 16-bit PCM (-32768 to 32767)
        with proper scaling and range checking.
        
        Args:
            adc_value: ADC sample value (should be 0-1023)
            
        Note:
            Out-of-range values are clipped to valid range with a warning.
            DC offset removal uses a running average if enabled.
        """
        if not 0 <= adc_value <= 1023:
            print(f"⚠️ Warning: ADC value {adc_value} out of range! Clipping...")
            adc_value = max(0, min(adc_value, 1023))

        # Convert to float for processing
        pcm_float = ((adc_value / 1023.0) * 65535.0) - 32768.0
        
        # Update DC offset estimate
        if self.remove_dc:
            alpha = 0.001  # Slow tracking for DC
            self.dc_offset = (alpha * pcm_float + 
                            (1 - alpha) * self.dc_offset)
            pcm_float -= self.dc_offset
        
        # Apply compression if enabled
        pcm_float = self._apply_compression(pcm_float)
        
        # Convert to integer and clip
        pcm_value = int(np.clip(pcm_float, -32768, 32767))
        
        # Update statistics
        self._update_stats(pcm_value)
        
        # Add to buffer
        self.buffer.append(pcm_value)
        
        # Process buffer when full
        if len(self.buffer) >= self.buffer_size:
            self.samples.extend(self.buffer)
            self.buffer.clear()

    def get_stats(self) -> Dict[str, float]:
        """
        Get current audio statistics.
        
        Returns:
            Dictionary containing:
            - min: Minimum sample value
            - max: Maximum sample value
            - mean: Average sample value
            - rms: Root mean square value
            - peak: Peak absolute value
            - peak_db: Peak level in dB
        """
        if self.stats['count'] == 0:
            return {}
            
        mean = self.stats['sum'] / self.stats['count']
        rms = np.sqrt(self.stats['sum_squared'] / self.stats['count'])
        
        return {
            'min': self.stats['min'],
            'max': self.stats['max'],
            'mean': mean,
            'rms': rms,
            'peak': self.peak_value,
            'peak_db': 20 * np.log10(self.peak_value / 32768.0)
        }

    def save_to_file(self) -> None:
        """
        Write collected samples to WAV file.
        
        Creates a standard WAV file with proper headers and writes all
        collected samples. Uses 16-bit PCM format, mono channel.
        Also saves a metadata file with audio statistics.
        
        Note:
            Prints a warning if no samples were collected.
        """
        # Process any remaining buffered samples
        if self.buffer:
            self.samples.extend(self.buffer)
            self.buffer.clear()
            
        if not self.samples:
            print("⚠️ No audio data recorded. WAV file will be empty.")
            return

        # Save WAV file
        with wave.open(self.filename, "w") as wav_file:
            # Set WAV file parameters
            wav_file.setnchannels(1)        # Mono audio
            wav_file.setsampwidth(2)        # 16-bit PCM
            wav_file.setframerate(self.sample_rate)

            # Convert samples to bytes and write
            audio_data = np.array(self.samples, dtype=np.int16).tobytes()
            wav_file.writeframes(audio_data)

        # Save metadata
        stats = self.get_stats()
        metadata_file = f"{self.output_dir}/audio_metadata.txt"
        with open(metadata_file, "w") as f:
            f.write("Audio Recording Metadata\n")
            f.write("=======================\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sample Rate: {self.sample_rate} Hz\n")
            f.write(f"Duration: {len(self.samples)/self.sample_rate:.2f} seconds\n")
            f.write(f"Samples: {len(self.samples)}\n")
            f.write(f"DC Offset Removal: {self.remove_dc}\n")
            f.write(f"Compression Factor: {self.compression_factor}\n\n")
            f.write("Statistics:\n")
            for key, value in stats.items():
                f.write(f"  {key}: {value:.2f}\n")

        print(f"✅ Audio saved to {self.filename}")
        print(f"✅ Metadata saved to {metadata_file}")
        print(f"   Peak level: {stats.get('peak_db', 0):.1f} dB")
