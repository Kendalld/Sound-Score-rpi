#!/bin/bash

# Create test tones
echo "Generating test tones..."
sox -n /tmp/quiet.wav synth 3 sine 1000 vol 0.1
sox -n /tmp/medium.wav synth 3 sine 1000 vol 0.5
sox -n /tmp/loud.wav synth 3 sine 1000 vol 0.9

# Function to play sound and wait for input
play_and_measure() {
    local sound=$1
    local level=$2
    echo -e "\nPlaying ${level} level tone..."
    echo "Press Enter when ready to play the sound..."
    read
    play $sound
    echo "Note the variation value shown above and press Enter to continue..."
    read
}

# Instructions
echo "This script will help calibrate the noise scoring system."
echo "Make sure your speaker volume is at a comfortable level."
echo "We'll play three test tones: quiet, medium, and loud."
echo "Watch the noise score display and note the 'var' values."
echo -e "\nPosition your speaker at a typical distance from the microphone."

# Play calibration sounds
play_and_measure "/tmp/quiet.wav" "quiet"
play_and_measure "/tmp/medium.wav" "medium"
play_and_measure "/tmp/loud.wav" "loud"

# Now play some real-world samples
echo -e "\nNow let's try some real-world sounds..."

# Generate white noise for background noise simulation
echo -e "\nGenerating background noise simulation..."
sox -n /tmp/background.wav synth 3 whitenoise vol 0.1

echo "Press Enter to play simulated background noise..."
read
play /tmp/background.wav

# Clean up
rm /tmp/quiet.wav /tmp/medium.wav /tmp/loud.wav /tmp/background.wav

echo -e "\nCalibration complete!"
echo "Use the observed 'var' values to adjust these parameters in plotter.py:"
echo "1. self.baseline_floor (currently 5.0)"
echo "2. self.baseline_ceiling multiplier (currently 4x)"
echo -e "\nRecommended settings based on typical values:"
echo "- Set baseline_floor to the variation value from quiet tone"
echo "- Set baseline_ceiling to the variation value from loud tone" 