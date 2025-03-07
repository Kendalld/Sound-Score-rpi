"""
MCP3009 ADC Interface Module

This module provides a clean interface to the MCP3009 10-bit ADC over SPI.
The MCP3009 is a 10-bit ADC with 8 input channels, commonly used in embedded
systems for analog signal acquisition.

Features:
- 8 single-ended input channels
- 10-bit resolution (0-1023 range)
- Configurable SPI speed and mode
- Clean error handling and resource management
- Channel configuration and calibration
- Voltage reference support
"""

import spidev
import time
from typing import Optional, Dict, Union
import numpy as np

class MCP3009:
    """
    Interface for MCP3009 10-bit ADC.
    
    Provides methods to read analog values from any of the 8 input channels
    using SPI communication. Handles device initialization, reading, and cleanup.
    
    Attributes:
        spi (spidev.SpiDev): SPI interface object
        vref (float): Reference voltage for ADC conversion
        channel_offsets (dict): DC offset calibration for each channel
        channel_gains (dict): Gain calibration for each channel
        
    Note:
        The ADC provides 10-bit resolution, giving values from 0 to 1023.
        SPI speed should be set appropriately for reliable readings.
    """
    
    def __init__(self, bus: int = 0, device: int = 0, speed: int = 1350000,
                 vref: float = 3.3, mode: int = 0):
        """
        Initialize SPI communication with the ADC.
        
        Args:
            bus: SPI bus number (default: 0)
            device: SPI device/chip select (default: 0)
            speed: SPI clock frequency in Hz (default: 1.35MHz)
            vref: ADC reference voltage in volts (default: 3.3V)
            mode: SPI mode (0-3, default: 0)
            
        Note:
            SPI modes:
            - Mode 0: CPOL=0, CPHA=0 (default)
            - Mode 1: CPOL=0, CPHA=1
            - Mode 2: CPOL=1, CPHA=0
            - Mode 3: CPOL=1, CPHA=1
        """
        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = speed
        self.spi.mode = mode
        
        self.vref = vref
        self.channel_offsets = {i: 0.0 for i in range(8)}
        self.channel_gains = {i: 1.0 for i in range(8)}
        
        # Verify communication
        self._verify_communication()

    def _verify_communication(self) -> None:
        """
        Verify SPI communication with the ADC.
        
        Attempts to read from channel 0 to ensure the ADC is responding.
        Raises RuntimeError if communication fails.
        """
        try:
            value = self.read_channel(0)
            if value is None:
                raise RuntimeError("Failed to read from ADC")
        except Exception as e:
            raise RuntimeError(f"ADC communication failed: {str(e)}")

    def calibrate_channel(self, channel: int, known_voltage: float) -> None:
        """
        Calibrate a channel using a known reference voltage.
        
        Args:
            channel: Channel number (0-7)
            known_voltage: Known input voltage for calibration
            
        Note:
            This updates both offset and gain calibration for the channel.
        """
        if not 0 <= channel <= 7:
            raise ValueError("Channel must be between 0 and 7")
            
        # Take multiple readings and average
        readings = []
        for _ in range(10):
            value = self.read_raw(channel)
            if value is not None:
                readings.append(value)
            time.sleep(0.01)
            
        if not readings:
            raise RuntimeError(f"Failed to get readings from channel {channel}")
            
        # Calculate calibration
        avg_reading = np.mean(readings)
        expected_reading = (known_voltage / self.vref) * 1023
        
        self.channel_gains[channel] = expected_reading / avg_reading
        self.channel_offsets[channel] = 0  # Reset offset after gain calibration

    def read_raw(self, channel: int) -> Optional[int]:
        """
        Read raw ADC value without calibration.
        
        Args:
            channel: ADC channel number (0-7)
            
        Returns:
            Raw 10-bit ADC value (0-1023), or None if read fails
        """
        if not 0 <= channel <= 7:
            raise ValueError("Channel must be between 0 and 7")

        try:
            command = [1, (8 + channel) << 4, 0]
            reply = self.spi.xfer2(command)
            return ((reply[1] & 3) << 8) + reply[2]
            
        except Exception as e:
            print(f"\n⚠️ Warning: SPI read failed: {str(e)}")
            return None

    def read_channel(self, channel: int) -> Optional[int]:
        """
        Read calibrated value from specified ADC channel.
        
        Performs SPI communication to read a single analog value from
        the specified channel. Applies calibration and includes error checking.
        
        Args:
            channel: ADC channel number (0-7)
            
        Returns:
            Calibrated 10-bit ADC value (0-1023), or None if read fails
            
        Raises:
            ValueError: If channel number is invalid
        """
        raw_value = self.read_raw(channel)
        if raw_value is None:
            return None
            
        # Apply calibration
        calibrated = (raw_value * self.channel_gains[channel] + 
                     self.channel_offsets[channel])
        
        # Ensure value stays within ADC range
        return int(np.clip(calibrated, 0, 1023))

    def read_voltage(self, channel: int) -> Optional[float]:
        """
        Read voltage from specified channel.
        
        Converts ADC value to actual voltage using reference voltage
        and calibration data.
        
        Args:
            channel: ADC channel number (0-7)
            
        Returns:
            Voltage in volts, or None if read fails
        """
        value = self.read_channel(channel)
        if value is None:
            return None
            
        return (value / 1023.0) * self.vref

    def get_channel_info(self, channel: int) -> Dict[str, Union[int, float]]:
        """
        Get diagnostic information for a channel.
        
        Args:
            channel: ADC channel number (0-7)
            
        Returns:
            Dictionary containing:
            - raw_value: Raw ADC reading
            - calibrated_value: Calibrated ADC reading
            - voltage: Converted voltage
            - gain: Current gain calibration
            - offset: Current offset calibration
        """
        raw = self.read_raw(channel)
        if raw is None:
            return {}
            
        cal = self.read_channel(channel)
        vol = self.read_voltage(channel)
        
        return {
            'raw_value': raw,
            'calibrated_value': cal,
            'voltage': vol,
            'gain': self.channel_gains[channel],
            'offset': self.channel_offsets[channel]
        }

    def close(self) -> None:
        """
        Clean up SPI resources.
        
        Should be called when done with ADC readings to properly
        close the SPI connection.
        """
        try:
            self.spi.close()
        except Exception as e:
            print(f"⚠️ Warning: Error closing SPI: {str(e)}")
