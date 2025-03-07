"""
Sound Level Monitor - Main Script

This script records audio from two MCP3009 ADC channels and performs real-time
analysis of sound levels. Channel 0 is used for visualization, and Channel 1
is used for audio recording and noise scoring.

The program will:
1. Record audio at 8kHz sample rate
2. Generate a real-time noise score (1-100)
3. Save the recorded audio as a WAV file
4. Create visualizations of the sound levels
5. Log all data for later analysis

Usage:
    python main.py [--filename FILENAME]
"""

import time
import argparse
import numpy as np
from adc_reader import MCP3009
from plotter import LivePlot
from audio_writer import AudioWriter
from signal_processing import SignalProcessor

# Recording settings
SAMPLE_RATE = 8000      # Sample rate in Hz
BUFFER_SIZE = 800       # Samples to process at once (100ms worth of data)
PLOT_UPDATE_INTERVAL = 0.1  # Seconds between plot updates

def main(audio_filename):
    """
    Main recording and analysis loop.
    
    Args:
        audio_filename (str): Name of the output WAV file
    """
    # Initialize hardware and analysis components
    adc = MCP3009(speed=2000000)  # 2MHz SPI for better timing
    plotter = LivePlot(buffer_size=500, save_file="plot_data.csv")
    writer = AudioWriter(filename=audio_filename, sample_rate=SAMPLE_RATE)
    processor = SignalProcessor(sample_rate=SAMPLE_RATE, cutoff_freq=3000)

    print(f"Recording at {SAMPLE_RATE} Hz... Saving audio to {audio_filename}")
    print("Calibrating noise levels... Please maintain normal background noise...")

    # Initialize timing and buffers
    start_time = time.time()
    last_plot_time = start_time
    sample_buffer = []
    fft_buffer = []

    try:
        while True:
            # Read audio channel
            ch1_value = adc.read_channel(1)
            current_time = time.time()

            if ch1_value is None:
                print("\r⚠️  Bad sample, retrying...", end='', flush=True)
                continue

            # Process audio sample
            plotter.update_noise_score(ch1_value, current_time - start_time)
            sample_buffer.append(ch1_value)
            fft_buffer.append(ch1_value)

            # Update visualization (every 100ms)
            if current_time - last_plot_time >= PLOT_UPDATE_INTERVAL:
                ch0_value = adc.read_channel(0)
                if ch0_value is not None:
                    plotter.update_plot(ch0_value, current_time - start_time)
                last_plot_time = current_time

            # Write audio in chunks for better performance
            if len(sample_buffer) >= 100:
                for sample in sample_buffer:
                    writer.add_sample(sample)
                sample_buffer.clear()

            # Process FFT for frequency analysis
            if len(fft_buffer) >= BUFFER_SIZE:
                filtered_audio = processor.apply_low_pass_filter(np.array(fft_buffer))
                filtered_audio = filtered_audio - np.mean(filtered_audio)  # Remove DC bias
                processor.compute_fft(filtered_audio)
                fft_buffer.clear()

            # Maintain correct sampling rate
            time.sleep(max(0, 1.0/SAMPLE_RATE - 0.0001))

    except KeyboardInterrupt:
        print("\n\nStopping recording... Saving files.")

        # Save any remaining samples
        if sample_buffer:
            for sample in sample_buffer:
                writer.add_sample(sample)

        # Process any remaining FFT data
        if fft_buffer:
            filtered_audio = processor.apply_low_pass_filter(np.array(fft_buffer))
            filtered_audio = filtered_audio - np.mean(filtered_audio)
            processor.compute_fft(filtered_audio)

        # Save all data files
        writer.save_to_file()
        plotter.save_and_close()
        processor.save_fft_plot()
        processor.save_fft_data()
        adc.close()

        print("✅ Done. Data saved in 'data/' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record and analyze audio from MCP3009 ADC."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="audio_output.wav",
        help="Output WAV filename (default: audio_output.wav)"
    )
    args = parser.parse_args()
    
    main(args.filename)
