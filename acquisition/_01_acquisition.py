
"""
Real-Time Force and Image Acquisition Module

This module handles the synchronized acquisition of force sensor data and camera images
for tactile sensing experiments. It provides the core data collection functionality
with precise timestamp synchronization between force measurements and images.

Key components:
    - wait_for_click(): Manual trigger for starting acquisition. The idea is to press the button to start the movement sequence on the steppers. This makes the recording start at the same time as the movement sequence.
    - ForceAcquisition: Real-time 6-axis force sensor data collection via NI-DAQ
    - CameraAcquisition: Real-time image capture via Basler camera
    - FTSensorConverter: Converts raw voltage to calibrated force/torque values

Typical workflow:
    1. Set up your ForceAcquisition and CameraAcquisition objects
    2. Call wait_for_click() to manually trigger start
    3. Start both acquisitions with synchronized timestamps
    4. Collect data for specified duration
    5. Stop acquisitions and process collected data

Hardware requirements:
    - NI-DAQ device for force sensor (tested with USB-6218 or similar)
    - Basler camera (GigE or USB3 models)
    - 6-axis force/torque sensor with calibration matrix

Directory context:
    Raw acquisition data will be saved to: Code_Machine_learning_Robotics/[trial_folder]/
    Use _02_saving_data.py module to organize and save the collected data.

All timestamps are synchronized using a common start time for ground-truth alignment
between force measurements and images - essential for machine learning datasets.
"""

import time
import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader
import threading
from pynput.mouse import Listener
from pypylon import pylon
import cv2
import gc; gc.collect()


# -----------------------------------------------------------------------------
# Manual trigger for acquisition start
# -----------------------------------------------------------------------------
def wait_for_click():
    """
    Pause program execution until a mouse button is clicked anywhere on screen. The idea is to press the button to start the movement sequence on the steppers. This makes the recording start at the same time as the movement sequence.
    
    This provides manual control over when acquisition starts, which is useful for:
        - Ensuring all hardware is ready before data collection
        - Synchronizing with external equipment or processes  
        - Giving yourself time to prepare the experimental setup
        - Manual timing control for specific experimental protocols
    
    Usage:
        print("Click anywhere when ready to start acquisition...")
        wait_for_click()
        # Acquisition will start immediately after mouse click
    
    The function will block program execution until any mouse button is pressed.
    """
    def on_click(x, y, button, pressed):
        if pressed:
            return False  # Stop listener after the first click

    with Listener(on_click=on_click) as listener:
        listener.join()  # Wait for mouse click before continuing


# -----------------------------------------------------------------------------
# Force sensor acquisition utilities
# -----------------------------------------------------------------------------
class FTSensorConverter:
    """
    Convert raw force sensor voltage measurements to calibrated force/torque values.
    
    Force sensors output raw voltage signals that must be converted to meaningful
    force and torque measurements using a calibration matrix provided by the manufacturer.
    This class handles that conversion for 6-axis force/torque sensors.
    
    Parameters:
        calibration_matrix (6x6 array): Sensor-specific calibration matrix
                                       Usually provided by sensor manufacturer
                                       Maps 6 voltage channels to [Fx,Fy,Fz,Mx,My,Mz]
    
    Typical usage:
        calibration_matrix = [...] # Your 6x6 calibration matrix
        converter = FTSensorConverter(calibration_matrix)
        forces = converter.convert_bulk(raw_voltage_data, bias_values)
    """
    def __init__(self, calibration_matrix):
        self.calibration_matrix = np.array(calibration_matrix)
        if self.calibration_matrix.shape != (6, 6):
            raise ValueError("Calibration matrix must be 6x6.")

    def convert_bulk(self, raw_data, bias):
        """
        Convert a batch of raw sensor data to force/torque values, correcting for bias.
        """
        raw_data = np.array(raw_data)
        bias = np.array(bias)
        corrected_data = raw_data - bias
        forcetorque = np.dot(corrected_data, self.calibration_matrix.T)
        return forcetorque


class ForceAcquisition:
    """
    Real-time force sensor data acquisition using NI-DAQ hardware.
    
    This class handles continuous acquisition of 6-axis force/torque data at high frequency
    (typically 1000+ Hz) with precise timestamping. Data is collected in chunks via callbacks
    to ensure no samples are lost during high-speed acquisition.
    
    Parameters:
        device (str): NI-DAQ device name (e.g., "Dev1", "cDAQ1Mod1")
        channels (str): Channel specification (e.g., "ai0:5" for 6 channels)
        frequency (int): Sampling rate in Hz (e.g., 1000)
        buffer_size (int): Samples per callback (e.g., 100)
        calibration_matrix (6x6 array): Force sensor calibration matrix
    
    Key features:
        - Continuous acquisition with callback-based data collection
        - Automatic bias correction using initial samples
        - Synchronized timestamping with camera acquisition
        - Thread-safe data storage
        - Quality diagnostics (callback counting, sample verification)
    
    Typical usage:
        force_acq = ForceAcquisition("Dev1", "ai0:5", 1000, 100, calib_matrix)
        force_acq.common_start_time = time.perf_counter()  # Sync with camera
        force_acq.start_acquisition()
        # ... run for desired duration ...
        force_acq.stop_acquisition()
        raw_data, forces, timestamps = force_acq.process_data()
    """
    def __init__(self, device, channels, frequency, buffer_size, calibration_matrix):
        self.device = device
        self.channels = channels
        self.frequency = frequency
        self.buffer_size = buffer_size
        self.calibration_matrix = calibration_matrix
        self.data_buffer = []
        self.acquisition_running = False
        self.lock = threading.Lock()
        self.task = None
        self.callback_registered = False
        self.timestamps_force = []
        self.callback_count = 0
        self.common_start_time = None  # Set externally for ground-truth alignment

    def initialize_task(self):
        """
        Initialize NI-DAQ task and configure acquisition parameters.
        """
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(f"{self.device}/{self.channels}")
        self.task.timing.cfg_samp_clk_timing(
            rate=self.frequency,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self.buffer_size,
        )
        print("Active channels:", self.task.ai_channels.channel_names)

    def record_data_callback(self, task_handle, every_n_samples_event_type, num_samples, callback_data):
        """
        Callback function to read and store force data for each buffer event.
        """
        try:
            reader = AnalogMultiChannelReader(self.task.in_stream)
            raw_data = np.empty((len(self.task.ai_channels.channel_names), num_samples))
            reader.read_many_sample(raw_data, num_samples)

            with self.lock:
                self.data_buffer.append(raw_data.T)
            if self.common_start_time is None:
                raise RuntimeError("common_start_time must be set before starting acquisition.")
            self.timestamps_force.append(time.perf_counter() - self.common_start_time)
            self.callback_count += 1

        except Exception as e:
            print("Error in callback:", e)
        return 0

    def start_acquisition(self):
        """
        Start force acquisition and register callback for buffer events.
        """
        if not self.task:
            self.initialize_task()

        if not self.callback_registered:
            self.task.register_every_n_samples_acquired_into_buffer_event(
                self.buffer_size, self.record_data_callback
            )
            self.callback_registered = True

        self.acquisition_running = True
        self.task.start()

    def stop_acquisition(self):
        """
        Stop and close the NI-DAQ task.
        """
        self.acquisition_running = False
        if self.task:
            self.task.stop()
            self.task.close()

    def process_data(self):
        """
        Process acquired force data, apply bias correction, and convert to force/torque values.
        """
        print("Processing acquired data...")
        raw_data = np.vstack(self.data_buffer)
        bias = np.mean(raw_data[:self.buffer_size, :], axis=0)
        print(f'Bias set to {bias}')
        converter = FTSensorConverter(self.calibration_matrix)
        forces = converter.convert_bulk(raw_data, bias)
        print("Data processing complete.")
        return raw_data, forces, self.timestamps_force

# -----------------------------------------------------------------------------
# Camera acquisition utilities
# -----------------------------------------------------------------------------
class CameraAcquisition:
    """
    Real-time image acquisition using Basler cameras with frame loss detection.
    
    This class handles continuous image capture at specified frame rates with precise
    timestamping synchronized to force acquisition. Includes automatic detection and
    reporting of dropped frames for quality control.
    
    Parameters:
        frequency (int): Target frame rate in fps (e.g., 10, 20, 50)
    
    Key features:
        - Configurable frame rate and exposure settings
        - Automatic throughput limit optimization
        - Frame loss detection with gap analysis
        - Synchronized timestamping with force acquisition
        - Image format conversion (Basler → BGR for OpenCV compatibility)
        - Acquisition diagnostics and frame rate analysis
    
    Hardware support:
        - Basler GigE cameras (acA series, daA series)
        - Basler USB3 cameras (acA USB series)
        - Requires pypylon library and Basler Pylon SDK
    
    Typical usage:
        camera = CameraAcquisition(frequency=10)
        camera.common_start_time = time.perf_counter()  # Sync with force
        images, timestamps, first_time, lost_frames = camera.run(duration=30.0)
        lost_count = camera.analyze_results(30.0, actual_elapsed_time)
    
    Output:
        - List of captured images (numpy arrays)
        - Timestamp for each image (synchronized with force data)
        - Frame loss diagnostics for quality assessment
    """
    def __init__(self, frequency):
        self.frequency = frequency
        self.timestamps = []            # Relative timestamps (seconds)
        self.image_data = []            # List of acquired images
        self.first_image_time = None    # Time of first image relative to acquisition start
        self.first_timestamp = None     # Basler timestamp (ns) of first image
        self.lost_frame_indices = []    # Indices of skipped frames
        self.common_start_time = None   # Set externally for ground-truth alignment

        # Initialize Basler camera and image converter
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def configure_camera(self):
        """
        Configure camera acquisition parameters and disable throughput limits.
        """
        self.camera.Open()

        if self.camera.DeviceLinkThroughputLimitMode.GetValue() == "On":
            self.camera.DeviceLinkThroughputLimitMode.SetValue("Off")
            print("Device Link Throughput Limit disabled.")
        else:
            print("Device Link Throughput Limit already disabled")

        self.camera.AcquisitionFrameRateEnable.SetValue(True)
        self.camera.AcquisitionFrameRate.SetValue(self.frequency)
        print(f"Acquisition frequency set to {self.camera.AcquisitionFrameRate.GetValue()} fps.")

        self.camera.ExposureAuto.Value = "Off"
        self.camera.ExposureTime.Value = 2000.0
        self.camera.GainAuto.Value = "Off"
        self.camera.Gain.Value = 0

    def run(self, acquisition_time):
        """
        Acquire images for the specified duration, timestamp each image, and detect dropped frames.
        """
        if self.common_start_time is None:
            raise RuntimeError("common_start_time must be set before starting camera acquisition.")

        self.configure_camera()
        self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByUser)
        first_image_acquired = False
        test_counter = 0

        try:
            while time.perf_counter() - self.common_start_time <= acquisition_time:
                grab_result = self.camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                test_counter += 1

                if grab_result.GrabSucceeded():
                    if not first_image_acquired:
                        self.first_image_time = time.perf_counter() - self.common_start_time
                        first_image_acquired = True

                    img = self.converter.Convert(grab_result).GetArray()
                    self.image_data.append(img)

                    timestamp_relative = time.perf_counter() - self.common_start_time
                    self.timestamps.append(timestamp_relative)

                    # Detect skipped frames by comparing timestamp gaps
                    if len(self.timestamps) > 1:
                        frame_gap = self.timestamps[-1] - self.timestamps[-2]
                        expected_gap = 1 / self.frequency
                        tolerance = expected_gap * 1.5 # 50% tolerance

                        if frame_gap > tolerance:
                            lost_index = len(self.timestamps) - 1
                            self.lost_frame_indices.append(lost_index)
                            print(f"[!] Frame skipped between frame {lost_index - 1} and {lost_index} "
                                  f"(gap: {frame_gap:.4f}s, expected ~{expected_gap:.4f}s)")

                else:
                    print(f"[!] An image was skipped at cycle {test_counter}")

                grab_result.Release()

        except Exception as e:
            print(f"Error during camera acquisition: {e}")
        finally:
            self.camera.StopGrabbing()
            self.camera.Close()
            print("Camera closed.")

        return self.image_data, self.timestamps, self.first_image_time, self.lost_frame_indices

    def analyze_results(self, acquisition_time, elapsed_time):
        """
        Summarize camera acquisition results and report lost frames.
        """
        expected_images = int(self.frequency * acquisition_time)
        actual_images = len(self.image_data)
        actual_frame_rate = actual_images / elapsed_time if elapsed_time > 0 else 0

        print("\n[Camera Acquisition Summary]")
        print(f"Requested acquisition frequency: {self.frequency} fps")
        print(f"Expected number of images: {expected_images}")
        print(f"Captured images: {actual_images}")
        print(f"Actual mean acquisition frequency: {actual_frame_rate:.4f} fps")

        # Average interval between frames (based on timestamps)
        if len(self.timestamps) > 1:
            intervals = [self.timestamps[i + 1] - self.timestamps[i] for i in range(len(self.timestamps) - 1)]
            avg_interval = sum(intervals) / len(intervals)
            print(f"Average interval between frames: {avg_interval:.6f} s")

        # Calculate lost frames (never negative)
        lost_frames = max(0, expected_images - actual_images)
        if lost_frames > 0:
            print(f"[!] Warning: {lost_frames} frames were lost.")
        else:
            print("[OK] No frames were lost (based on count).")

        return lost_frames
        return lost_frames
