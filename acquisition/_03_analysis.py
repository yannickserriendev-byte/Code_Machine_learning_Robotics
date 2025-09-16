"""
Acquisition Quality Control and Diagnostics Module

This module provides comprehensive diagnostic tools to verify the integrity and quality
of your force and image acquisition data. Use these functions after acquisition to
ensure your data is complete, properly timed, and suitable for analysis or machine learning.

Key diagnostic functions:
    - check_timestamp_differences(): Detect timing irregularities in any timestamp sequence
    - run_callback_and_sample_diagnostics(): Verify force acquisition completeness

Why use these diagnostics?
    Real-time acquisition can suffer from timing issues, missed callbacks, dropped frames,
    or hardware synchronization problems. These tools help identify such issues before
    you invest time in analysis or training machine learning models on flawed data.

Typical workflow:
    1. Complete your acquisition and save data
    2. Run diagnostics on force acquisition system:
       callback_missed, sample_loss = run_callback_and_sample_diagnostics(...)
    3. Check timestamp consistency for images and force data:
       problems = check_timestamp_differences(timestamps, expected_interval)
    4. Address any identified issues before proceeding to analysis

Quality indicators to check:
    - All expected force sensor callbacks executed
    - No missing force samples (complete data coverage)
    - Regular timing intervals in timestamp sequences
    - Consistent frame rates and acquisition timing

Directory context:
    Use with data from: Code_Machine_learning_Robotics/[trial_folder]/
    These diagnostics help ensure data quality before moving to processing stages.
"""

def check_timestamp_differences(timestampstocheck, expected_difference):
    """
    Detect timing inconsistencies in timestamp sequences.
    
    This function identifies where timestamps deviate from the expected regular interval,
    which could indicate missed acquisitions or timing issues.
    
    Args:
        timestampstocheck (list or np.array): Sequence of timestamps (in seconds)
                                            Example: [0.0, 0.1, 0.2, 0.4, 0.5] for 10 fps
        expected_difference (float): Expected time between consecutive timestamps (in seconds)
                                   Example: 0.1 for 10 fps camera, 0.001 for 1000 Hz force sensor
    
    Returns:
        list: Indices where timing deviates from expected (indicates potential problems)
              Example: [2] would mean problem between timestamps[2] and timestamps[3]
    
    Example:
        # For 10 fps camera, expect 0.1s between frames
        problem_indices = check_timestamp_differences(image_timestamps, 0.1)
        if problem_indices:
            print(f"Timing issues detected at frame transitions: {problem_indices}")
    """
    # Check each consecutive pair of timestamps for timing deviations
    differing_indices = []
    for i in range(len(timestampstocheck) - 1):
        diff = round(timestampstocheck[i + 1] - timestampstocheck[i], 4)
        if abs(diff - expected_difference) > expected_difference * 0.1:  # 10% tolerance
            differing_indices.append(i)
    return differing_indices

def run_callback_and_sample_diagnostics(acquisition, raw_data, buffer_size, force_acquisition_frequency, acquisition_time):
    """
    Comprehensive diagnostics for force acquisition system integrity.
    
    This function verifies two critical aspects of force data collection:
    1. Callback completeness: Did all expected NI-DAQ callbacks execute?
    2. Sample completeness: Were all expected force samples actually collected?
    
    Missing callbacks or samples indicate hardware/software timing issues that could
    compromise data quality for machine learning or analysis.

    Parameters:
        acquisition: ForceAcquisition object with .callback_count attribute
                    (the force acquisition instance that was used)
        raw_data: numpy.array with shape (total_samples, 6) containing raw force data
                 (the stacked force measurements from all callbacks)
        buffer_size: int, number of samples collected per callback (e.g., 100)
        force_acquisition_frequency: int, force sensor sampling rate in Hz (e.g., 1000)
        acquisition_time: float, total acquisition duration in seconds (e.g., 30.0)

    Returns:
        tuple: (callback_missed: bool, force_sample_loss: bool)
               - callback_missed: True if fewer callbacks than expected occurred
               - force_sample_loss: True if fewer samples than expected were collected
    
    Example:
        # After 30-second acquisition at 1000 Hz with 100 samples per callback
        callback_missed, sample_loss = run_callback_and_sample_diagnostics(
            force_acq_obj, raw_force_array, 100, 1000, 30.0
        )
        if callback_missed:
            print("Warning: Some force data callbacks were missed!")
        if sample_loss:
            print("Warning: Some force samples were lost!")
    """
    # Calculate expected vs actual callback counts
    expected_callbacks = int(acquisition_time * force_acquisition_frequency / buffer_size)
    actual_callbacks = acquisition.callback_count
    callback_missed = actual_callbacks < expected_callbacks

    print("\n[CALLBACK DIAGNOSTICS]")
    print(f"Expected callbacks: {expected_callbacks}")
    print(f"Actual callbacks:   {actual_callbacks}")
    if callback_missed:
        print(f"[!] Missed {expected_callbacks - actual_callbacks} callback(s).")
    else:
        print("[OK] All expected callbacks were executed.")

    # Calculate expected vs actual sample counts
    expected_samples = int(acquisition_time * force_acquisition_frequency)
    actual_samples = raw_data.shape[0]
    force_sample_loss = actual_samples < expected_samples

    print("\n[FORCE SAMPLE DIAGNOSTICS]")
    print(f"Expected total force samples: {expected_samples}")
    print(f"Actual force samples acquired: {actual_samples}")
    if force_sample_loss:
        print(f"[!] Lost approximately {expected_samples - actual_samples} force samples.")
    else:
        print("[OK] All expected force samples were collected.")

    return callback_missed, force_sample_loss
