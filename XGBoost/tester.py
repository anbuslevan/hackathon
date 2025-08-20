import json
import os

from kit import ModelMonitor, ProductionPredictor
from main import BASE_MODEL_DIR

# Example usage and testing
def example_usage():    
    # Example telemetry data
    sample_telemetry = {
        "record_id": "rec_001",
        "device_id": "device_12345",
        "user_id": "user_789",
        "timestamp": "2025-08-19T10:30:00",
        "device_model": "Samsung-Galaxy-S22",
        "device_manufacturer": "Samsung",
        "device_batch_number": "BATCH_2023_Q1_001",
        "device_production_date": "2023-01-15",
        "device_age_days": 581,
        "warranty_status": "in_warranty",
        "os_name": "Android",
        "os_version": "13",
        "os_build_number": "TP1A.220624.014",
        "software_firmware_version": "G996BXXU5EWBH",
        "software_security_patch": "2024-07-01",
        "software_uptime_hours": 72.5,
        "software_system_crashes": 0,
        "software_app_crashes": 2,
        "battery_level_percent": 45.2,
        "battery_health_status": "good",
        "battery_temperature_c": 32.1,
        "battery_voltage_mv": 3847,
        "battery_charge_cycles": 567,
        "battery_charging_status": "discharging",
        "battery_current_ma": -1250,
        "battery_resistance_mohm": 150,
        "battery_sudden_shutdowns": 0,
        "cpu_usage_percent": 23.4,
        "cpu_temperature_c_avg": 41.2,
        "cpu_throttle_events": 1,
        "cpu_core_frequencies_mhz": "1785,1785,2265,2265,2840,2840,2840,2840",
        "cpu_voltage_scaling_errors": 0,
        "cpu_watchdog_resets": 0,
        "gpu_usage_percent": 12.1,
        "gpu_temperature_c": 38.5,
        "gpu_driver_resets": 0,
        "gpu_frequency_mhz": 572,
        "memory_total_mb": 8192,
        "memory_used_mb": 4856,
        "memory_swap_used_mb": 0,
        "memory_page_faults": 1250,
        "memory_oom_kills": 0,
        "memory_ecc_corrected_errors": 0,
        "memory_dma_faults": 0,
        "storage_total_gb": 128,
        "storage_used_gb": 89,
        "storage_read_errors": 0,
        "storage_write_errors": 0,
        "storage_bad_block_count": 0,
        "storage_io_latency_ms": 4.2,
        "storage_wear_level_percent": 12.3,
        "thermal_hotspot_temp_c": 43.1,
        "thermal_sensor_battery_c": 32.1,
        "thermal_sensor_cpu_cluster0_c": 41.2,
        "thermal_sensor_cpu_cluster1_c": 42.8,
        "thermal_sensor_gpu_c": 38.5,
        "thermal_sensor_pmic_c": 35.7,
        "thermal_shutdowns": 0,
        "wifi_signal_dbm": -67,
        "wifi_disconnects": 2,
        "wifi_firmware_crashes": 0,
        "wifi_packet_loss_percent": 0.8,
        "cellular_rsrp_dbm": -95,
        "cellular_drop_calls": 0,
        "cellular_modem_resets": 0,
        "cellular_handshake_failures": 1,
        "peripheral_touchscreen_errors": 0,
        "peripheral_camera_init_failures": 0,
        "peripheral_mic_speaker_faults": 0,
        "peripheral_sensor_hub_desyncs": 0,
        "kernel_panics": 0,
        "kernel_anrs": 1,
        "kernel_irq_storms": 0,
        "kernel_driver_probe_failures": 0,
        "kernel_system_crash_count": 0,
        "kernel_last_boot_reason": "reboot",
        "user_region": "US",
        "user_account_age_days": 1825,
        "user_device_count": 3,
        "user_failure_history_count": 1,
        "user_behavior_profile": "moderate_user",
        "failure_occurred": False,
        "failure_type": "none",
        "failure_timestamp": None
    }
    
    battery_failure_telemetry = {
        "record_id": "rec_battery_fail_001",
        "device_id": "device_battery_fail_001",
        "user_id": "user_heavy_001",
        "timestamp": "2025-08-19T14:30:00",
        "device_model": "Samsung-Galaxy-S22",
        "device_manufacturer": "Samsung",
        "device_batch_number": "BATCH_2022_Q2_005",
        "device_production_date": "2022-04-15",
        "device_age_days": 856,  # Old device
        "warranty_status": "expired",
        "os_name": "Android",
        "os_version": "13",
        "os_build_number": "TP1A.220624.014",
        "software_firmware_version": "G996BXXU5EWBH",
        "software_security_patch": "2024-07-01",
        "software_uptime_hours": 120.5,
        "software_system_crashes": 3,
        "software_app_crashes": 8,
        "battery_level_percent": 15.2,  # Low battery
        "battery_health_status": "poor",  # Poor health
        "battery_temperature_c": 52.3,  # OVERHEATING
        "battery_voltage_mv": 3245,  # Low voltage
        "battery_charge_cycles": 1850,  # High cycles
        "battery_charging_status": "discharging",
        "battery_current_ma": -2850,  # High drain
        "battery_resistance_mohm": 450,  # High resistance - BAD
        "battery_sudden_shutdowns": 8,  # Multiple shutdowns
        "cpu_usage_percent": 85.4,
        "cpu_temperature_c_avg": 68.2,  # Hot CPU
        "cpu_throttle_events": 15,  # Frequent throttling
        "cpu_core_frequencies_mhz": "1200,1200,1800,1800,2200,2200,2200,2200",
        "cpu_voltage_scaling_errors": 3,
        "cpu_watchdog_resets": 1,
        "gpu_usage_percent": 95.1,
        "gpu_temperature_c": 72.5,
        "gpu_driver_resets": 2,
        "gpu_frequency_mhz": 572,
        "memory_total_mb": 8192,
        "memory_used_mb": 7856,  # High memory usage
        "memory_swap_used_mb": 512,
        "memory_page_faults": 8500,
        "memory_oom_kills": 3,
        "memory_ecc_corrected_errors": 0,
        "memory_dma_faults": 0,
        "storage_total_gb": 128,
        "storage_used_gb": 120,  # Almost full
        "storage_read_errors": 2,
        "storage_write_errors": 1,
        "storage_bad_block_count": 0,
        "storage_io_latency_ms": 45.2,  # High latency
        "storage_wear_level_percent": 85.3,  # High wear
        "thermal_hotspot_temp_c": 72.1,  # CRITICAL TEMPERATURE
        "thermal_sensor_battery_c": 52.3,
        "thermal_sensor_cpu_cluster0_c": 68.2,
        "thermal_sensor_cpu_cluster1_c": 70.8,
        "thermal_sensor_gpu_c": 72.5,
        "thermal_sensor_pmic_c": 65.7,
        "thermal_shutdowns": 2,  # Thermal shutdowns occurred
        "wifi_signal_dbm": -85,
        "wifi_disconnects": 12,
        "wifi_firmware_crashes": 1,
        "wifi_packet_loss_percent": 8.5,
        "cellular_rsrp_dbm": -110,
        "cellular_drop_calls": 3,
        "cellular_modem_resets": 2,
        "cellular_handshake_failures": 8,
        "peripheral_touchscreen_errors": 0,
        "peripheral_camera_init_failures": 1,
        "peripheral_mic_speaker_faults": 0,
        "peripheral_sensor_hub_desyncs": 2,
        "kernel_panics": 1,
        "kernel_anrs": 5,
        "kernel_irq_storms": 2,
        "kernel_driver_probe_failures": 1,
        "kernel_system_crash_count": 3,
        "kernel_last_boot_reason": "thermal_shutdown",
        "user_region": "US",
        "user_account_age_days": 1825,
        "user_device_count": 3,
        "user_failure_history_count": 2,  # History of failures
        "user_behavior_profile": "heavy_gamer",
        "failure_occurred": True,
        "failure_type": "battery",
        "failure_timestamp": "2025-08-19T14:30:00"
    }
    
    cpu_failure_telemetry = {
        "record_id": "rec_cpu_fail_001",
        "device_id": "device_cpu_fail_001", 
        "user_id": "user_heavy_002",
        "timestamp": "2025-08-19T16:45:00",
        "device_model": "Samsung-Galaxy-S22",
        "device_manufacturer": "Samsung",
        "device_batch_number": "BATCH_2022_Q3_012",
        "device_production_date": "2022-07-20",
        "device_age_days": 760,
        "warranty_status": "expired",
        "os_name": "Android", 
        "os_version": "13",
        "os_build_number": "TP1A.220624.014",
        "software_firmware_version": "G996BXXU5EWBH",
        "software_security_patch": "2024-07-01",
        "software_uptime_hours": 96.8,
        "software_system_crashes": 12,  # Frequent crashes
        "software_app_crashes": 25,
        "battery_level_percent": 35.2,
        "battery_health_status": "fair",
        "battery_temperature_c": 45.8,  # Warm from CPU heat
        "battery_voltage_mv": 3654,
        "battery_charge_cycles": 1245,
        "battery_charging_status": "discharging", 
        "battery_current_ma": -3250,  # High drain due to CPU
        "battery_resistance_mohm": 285,
        "battery_sudden_shutdowns": 4,
        "cpu_usage_percent": 98.7,  # MAXED OUT
        "cpu_temperature_c_avg": 89.5,  # CRITICAL CPU TEMPERATURE
        "cpu_throttle_events": 45,  # EXCESSIVE THROTTLING
        "cpu_core_frequencies_mhz": "800,800,1200,1200,1600,1600,1600,1600",  # Throttled down
        "cpu_voltage_scaling_errors": 12,  # VOLTAGE PROBLEMS
        "cpu_watchdog_resets": 5,  # Frequent resets
        "gpu_usage_percent": 45.1,
        "gpu_temperature_c": 68.2,
        "gpu_driver_resets": 1,
        "gpu_frequency_mhz": 400,  # Throttled
        "memory_total_mb": 8192,
        "memory_used_mb": 7950,
        "memory_swap_used_mb": 1024,  # High swap usage
        "memory_page_faults": 15000,
        "memory_oom_kills": 8,  # Many killed processes
        "memory_ecc_corrected_errors": 2,
        "memory_dma_faults": 1,
        "storage_total_gb": 128,
        "storage_used_gb": 115,
        "storage_read_errors": 0,
        "storage_write_errors": 0,
        "storage_bad_block_count": 0,
        "storage_io_latency_ms": 25.6,
        "storage_wear_level_percent": 45.8,
        "thermal_hotspot_temp_c": 91.2,  # EXTREME HOTSPOT
        "thermal_sensor_battery_c": 45.8,
        "thermal_sensor_cpu_cluster0_c": 89.5,
        "thermal_sensor_cpu_cluster1_c": 91.2,
        "thermal_sensor_gpu_c": 68.2,
        "thermal_sensor_pmic_c": 75.3,
        "thermal_shutdowns": 8,  # Multiple thermal shutdowns
        "wifi_signal_dbm": -72,
        "wifi_disconnects": 5,
        "wifi_firmware_crashes": 0,
        "wifi_packet_loss_percent": 2.1,
        "cellular_rsrp_dbm": -98,
        "cellular_drop_calls": 1,
        "cellular_modem_resets": 0,
        "cellular_handshake_failures": 3,
        "peripheral_touchscreen_errors": 2,  # Heat affecting touch
        "peripheral_camera_init_failures": 3,  # Heat issues
        "peripheral_mic_speaker_faults": 1,
        "peripheral_sensor_hub_desyncs": 5,
        "kernel_panics": 8,  # Frequent kernel panics
        "kernel_anrs": 15,
        "kernel_irq_storms": 6,
        "kernel_driver_probe_failures": 4,
        "kernel_system_crash_count": 12,
        "kernel_last_boot_reason": "kernel_panic",
        "user_region": "US",
        "user_account_age_days": 2190,
        "user_device_count": 2,
        "user_failure_history_count": 3,
        "user_behavior_profile": "heavy_gamer",
        "failure_occurred": True,
        "failure_type": "cpu",
        "failure_timestamp": "2025-08-19T16:45:00"
    }
    
    storage_failure_telemetry = {
        "record_id": "rec_storage_fail_001",
        "device_id": "device_storage_fail_001",
        "user_id": "user_moderate_003",
        "timestamp": "2025-08-19T11:15:00",
        "device_model": "Samsung-Galaxy-S22",
        "device_manufacturer": "Samsung", 
        "device_batch_number": "BATCH_2022_Q1_008",
        "device_production_date": "2022-02-10",
        "device_age_days": 923,  # Old device
        "warranty_status": "expired",
        "os_name": "Android",
        "os_version": "13",
        "os_build_number": "TP1A.220624.014",
        "software_firmware_version": "G996BXXU5EWBH",
        "software_security_patch": "2024-07-01",
        "software_uptime_hours": 48.2,
        "software_system_crashes": 8,
        "software_app_crashes": 18,
        "battery_level_percent": 67.8,
        "battery_health_status": "fair",
        "battery_temperature_c": 38.5,
        "battery_voltage_mv": 3789,
        "battery_charge_cycles": 1456,
        "battery_charging_status": "discharging",
        "battery_current_ma": -1850,
        "battery_resistance_mohm": 325,
        "battery_sudden_shutdowns": 2,
        "cpu_usage_percent": 45.6,
        "cpu_temperature_c_avg": 52.1,
        "cpu_throttle_events": 3,
        "cpu_core_frequencies_mhz": "1785,1785,2265,2265,2600,2600,2600,2600",
        "cpu_voltage_scaling_errors": 0,
        "cpu_watchdog_resets": 0,
        "gpu_usage_percent": 28.4,
        "gpu_temperature_c": 48.2,
        "gpu_driver_resets": 0,
        "gpu_frequency_mhz": 572,
        "memory_total_mb": 8192,
        "memory_used_mb": 5824,
        "memory_swap_used_mb": 256,
        "memory_page_faults": 4500,
        "memory_oom_kills": 1,
        "memory_ecc_corrected_errors": 0,
        "memory_dma_faults": 0,
        "storage_total_gb": 128,
        "storage_used_gb": 98,
        "storage_read_errors": 45,  # HIGH READ ERRORS
        "storage_write_errors": 32,  # HIGH WRITE ERRORS
        "storage_bad_block_count": 8,  # BAD BLOCKS DETECTED
        "storage_io_latency_ms": 125.6,  # VERY HIGH LATENCY
        "storage_wear_level_percent": 92.4,  # CRITICAL WEAR LEVEL
        "thermal_hotspot_temp_c": 55.8,
        "thermal_sensor_battery_c": 38.5,
        "thermal_sensor_cpu_cluster0_c": 52.1,
        "thermal_sensor_cpu_cluster1_c": 53.7,
        "thermal_sensor_gpu_c": 48.2,
        "thermal_sensor_pmic_c": 45.1,
        "thermal_shutdowns": 0,
        "wifi_signal_dbm": -75,
        "wifi_disconnects": 3,
        "wifi_firmware_crashes": 0,
        "wifi_packet_loss_percent": 1.2,
        "cellular_rsrp_dbm": -102,
        "cellular_drop_calls": 0,
        "cellular_modem_resets": 0,
        "cellular_handshake_failures": 2,
        "peripheral_touchscreen_errors": 0,
        "peripheral_camera_init_failures": 2,  # Storage access issues
        "peripheral_mic_speaker_faults": 0,
        "peripheral_sensor_hub_desyncs": 1,
        "kernel_panics": 3,  # Some panics from I/O errors
        "kernel_anrs": 8,
        "kernel_irq_storms": 1,
        "kernel_driver_probe_failures": 2,
        "kernel_system_crash_count": 8,
        "kernel_last_boot_reason": "reboot",
        "user_region": "EU",
        "user_account_age_days": 2556,
        "user_device_count": 4,
        "user_failure_history_count": 1,
        "user_behavior_profile": "moderate_user",
        "failure_occurred": True,
        "failure_type": "storage",
        "failure_timestamp": "2025-08-19T11:15:00"
    }

    memory_failure_telemetry = {
        "record_id": "rec_memory_fail_001",
        "device_id": "device_memory_fail_001",
        "user_id": "user_power_004", 
        "timestamp": "2025-08-19T09:20:00",
        "device_model": "Samsung-Galaxy-S22",
        "device_manufacturer": "Samsung",
        "device_batch_number": "BATCH_2022_Q4_015",
        "device_production_date": "2022-10-05",
        "device_age_days": 683,
        "warranty_status": "in_warranty",
        "os_name": "Android",
        "os_version": "13", 
        "os_build_number": "TP1A.220624.014",
        "software_firmware_version": "G996BXXU5EWBH",
        "software_security_patch": "2024-07-01",
        "software_uptime_hours": 24.1,  # Frequent reboots
        "software_system_crashes": 15,  # Many crashes
        "software_app_crashes": 32,
        "battery_level_percent": 78.2,
        "battery_health_status": "good",
        "battery_temperature_c": 36.8,
        "battery_voltage_mv": 3912,
        "battery_charge_cycles": 845,
        "battery_charging_status": "charging",
        "battery_current_ma": 1650,
        "battery_resistance_mohm": 185,
        "battery_sudden_shutdowns": 6,  # Memory-related shutdowns
        "cpu_usage_percent": 72.3,
        "cpu_temperature_c_avg": 58.7,
        "cpu_throttle_events": 8,
        "cpu_core_frequencies_mhz": "1785,1785,2265,2265,2840,2840,2840,2840",
        "cpu_voltage_scaling_errors": 1,
        "cpu_watchdog_resets": 2,
        "gpu_usage_percent": 65.8,
        "gpu_temperature_c": 55.2,
        "gpu_driver_resets": 3,  # GPU issues due to memory
        "gpu_frequency_mhz": 572,
        "memory_total_mb": 8192,
        "memory_used_mb": 8100,  # ALMOST FULL
        "memory_swap_used_mb": 2048,  # HEAVY SWAP USAGE
        "memory_page_faults": 25000,  # EXCESSIVE PAGE FAULTS
        "memory_oom_kills": 18,  # FREQUENT OOM KILLS
        "memory_ecc_corrected_errors": 15,  # ECC ERRORS DETECTED
        "memory_dma_faults": 6,  # DMA FAULTS
        "storage_total_gb": 128,
        "storage_used_gb": 95,
        "storage_read_errors": 3,
        "storage_write_errors": 8,  # High due to swap
        "storage_bad_block_count": 1,
        "storage_io_latency_ms": 85.3,  # High due to memory pressure
        "storage_wear_level_percent": 68.7,
        "thermal_hotspot_temp_c": 62.4,
        "thermal_sensor_battery_c": 36.8,
        "thermal_sensor_cpu_cluster0_c": 58.7,
        "thermal_sensor_cpu_cluster1_c": 60.2,
        "thermal_sensor_gpu_c": 55.2,
        "thermal_sensor_pmic_c": 48.9,
        "thermal_shutdowns": 1,
        "wifi_signal_dbm": -68,
        "wifi_disconnects": 4,
        "wifi_firmware_crashes": 1,  # Memory issues affecting WiFi
        "wifi_packet_loss_percent": 3.2,
        "cellular_rsrp_dbm": -92,
        "cellular_drop_calls": 2,
        "cellular_modem_resets": 1,
        "cellular_handshake_failures": 5,
        "peripheral_touchscreen_errors": 1,
        "peripheral_camera_init_failures": 4,  # Memory allocation failures
        "peripheral_mic_speaker_faults": 2,
        "peripheral_sensor_hub_desyncs": 8,  # Memory sync issues
        "kernel_panics": 12,  # Memory-related panics
        "kernel_anrs": 20,
        "kernel_irq_storms": 8,
        "kernel_driver_probe_failures": 6,
        "kernel_system_crash_count": 15,
        "kernel_last_boot_reason": "kernel_panic",
        "user_region": "ASIA",
        "user_account_age_days": 1095,
        "user_device_count": 5,
        "user_failure_history_count": 2,
        "user_behavior_profile": "power_user",
        "failure_occurred": True,
        "failure_type": "memory",
        "failure_timestamp": "2025-08-19T09:20:00"
    }

    # 5. CONNECTIVITY FAILURE - WiFi/Cellular issues
    connectivity_failure_telemetry = {
        "record_id": "rec_connectivity_fail_001",
        "device_id": "device_connectivity_fail_001",
        "user_id": "user_mobile_005",
        "timestamp": "2025-08-19T13:40:00", 
        "device_model": "Samsung-Galaxy-S22",
        "device_manufacturer": "Samsung",
        "device_batch_number": "BATCH_2023_Q2_022",
        "device_production_date": "2023-05-12",
        "device_age_days": 464,
        "warranty_status": "in_warranty",
        "os_name": "Android",
        "os_version": "13",
        "os_build_number": "TP1A.220624.014", 
        "software_firmware_version": "G996BXXU5EWBH",
        "software_security_patch": "2024-07-01",
        "software_uptime_hours": 156.8,
        "software_system_crashes": 2,
        "software_app_crashes": 6,
        "battery_level_percent": 42.1,
        "battery_health_status": "good",
        "battery_temperature_c": 39.2,
        "battery_voltage_mv": 3768,
        "battery_charge_cycles": 432,
        "battery_charging_status": "discharging",
        "battery_current_ma": -2150,  # High drain searching for signal
        "battery_resistance_mohm": 165,
        "battery_sudden_shutdowns": 0,
        "cpu_usage_percent": 65.8,  # High usage from connectivity attempts
        "cpu_temperature_c_avg": 48.9,
        "cpu_throttle_events": 2,
        "cpu_core_frequencies_mhz": "1785,1785,2265,2265,2840,2840,2840,2840",
        "cpu_voltage_scaling_errors": 0,
        "cpu_watchdog_resets": 0,
        "gpu_usage_percent": 18.5,
        "gpu_temperature_c": 42.1,
        "gpu_driver_resets": 0,
        "gpu_frequency_mhz": 572,
        "memory_total_mb": 8192,
        "memory_used_mb": 6245,
        "memory_swap_used_mb": 128,
        "memory_page_faults": 3200,
        "memory_oom_kills": 0,
        "memory_ecc_corrected_errors": 0,
        "memory_dma_faults": 0,
        "storage_total_gb": 128,
        "storage_used_gb": 78,
        "storage_read_errors": 0,
        "storage_write_errors": 0,
        "storage_bad_block_count": 0,
        "storage_io_latency_ms": 12.4,
        "storage_wear_level_percent": 28.6,
        "thermal_hotspot_temp_c": 52.3,
        "thermal_sensor_battery_c": 39.2,
        "thermal_sensor_cpu_cluster0_c": 48.9,
        "thermal_sensor_cpu_cluster1_c": 50.1,
        "thermal_sensor_gpu_c": 42.1,
        "thermal_sensor_pmic_c": 44.7,
        "thermal_shutdowns": 0,
        "wifi_signal_dbm": -95,  # VERY WEAK SIGNAL
        "wifi_disconnects": 45,  # FREQUENT DISCONNECTS
        "wifi_firmware_crashes": 8,  # WIFI FIRMWARE ISSUES
        "wifi_packet_loss_percent": 25.8,  # HIGH PACKET LOSS
        "cellular_rsrp_dbm": -115,  # VERY POOR CELLULAR
        "cellular_drop_calls": 12,  # FREQUENT CALL DROPS
        "cellular_modem_resets": 15,  # MODEM INSTABILITY
        "cellular_handshake_failures": 38,  # CONNECTION ISSUES
        "peripheral_touchscreen_errors": 0,
        "peripheral_camera_init_failures": 0,
        "peripheral_mic_speaker_faults": 3,  # Call quality issues
        "peripheral_sensor_hub_desyncs": 1,
        "kernel_panics": 1,
        "kernel_anrs": 4,
        "kernel_irq_storms": 3,  # Interrupt issues from connectivity
        "kernel_driver_probe_failures": 5,  # Radio driver issues
        "kernel_system_crash_count": 2,
        "kernel_last_boot_reason": "reboot",
        "user_region": "US",
        "user_account_age_days": 892,
        "user_device_count": 2,
        "user_failure_history_count": 0,
        "user_behavior_profile": "mobile_heavy",
        "failure_occurred": True,
        "failure_type": "connectivity",
        "failure_timestamp": "2025-08-19T13:40:00"
    }

    # 6. SYSTEM FAILURE - Kernel panics, multiple subsystem failures
    system_failure_telemetry = {
        "record_id": "rec_system_fail_001",
        "device_id": "device_system_fail_001",
        "user_id": "user_unlucky_006",
        "timestamp": "2025-08-19T18:55:00",
        "device_model": "Samsung-Galaxy-S22",
        "device_manufacturer": "Samsung",
        "device_batch_number": "BATCH_2022_Q2_003",  # Problematic batch
        "device_production_date": "2022-04-08",
        "device_age_days": 863,
        "warranty_status": "expired",
        "os_name": "Android",
        "os_version": "13",
        "os_build_number": "TP1A.220624.014",
        "software_firmware_version": "G996BXXU5EWBH",
        "software_security_patch": "2024-07-01",
        "software_uptime_hours": 8.2,  # Can't stay up long
        "software_system_crashes": 25,  # EXCESSIVE CRASHES
        "software_app_crashes": 48,
        "battery_level_percent": 28.7,
        "battery_health_status": "poor",
        "battery_temperature_c": 47.8,
        "battery_voltage_mv": 3456,
        "battery_charge_cycles": 1678,
        "battery_charging_status": "discharging",
        "battery_current_ma": -2950,
        "battery_resistance_mohm": 380,
        "battery_sudden_shutdowns": 12,  # Frequent shutdowns
        "cpu_usage_percent": 88.9,
        "cpu_temperature_c_avg": 75.6,  # High temperature
        "cpu_throttle_events": 28,
        "cpu_core_frequencies_mhz": "1200,1200,1600,1600,2000,2000,2000,2000",  # Throttled
        "cpu_voltage_scaling_errors": 8,
        "cpu_watchdog_resets": 12,  # Multiple watchdog resets
        "gpu_usage_percent": 76.4,
        "gpu_temperature_c": 71.8,
        "gpu_driver_resets": 6,
        "gpu_frequency_mhz": 400,  # Throttled
        "memory_total_mb": 8192,
        "memory_used_mb": 7890,
        "memory_swap_used_mb": 1536,
        "memory_page_faults": 18000,
        "memory_oom_kills": 12,
        "memory_ecc_corrected_errors": 8,
        "memory_dma_faults": 4,
        "storage_total_gb": 128,
        "storage_used_gb": 122,  # Almost full
        "storage_read_errors": 18,
        "storage_write_errors": 12,
        "storage_bad_block_count": 3,
        "storage_io_latency_ms": 95.7,
        "storage_wear_level_percent": 88.4,
        "thermal_hotspot_temp_c": 78.9,  # CRITICAL
        "thermal_sensor_battery_c": 47.8,
        "thermal_sensor_cpu_cluster0_c": 75.6,
        "thermal_sensor_cpu_cluster1_c": 78.9,
        "thermal_sensor_gpu_c": 71.8,
        "thermal_sensor_pmic_c": 68.2,
        "thermal_shutdowns": 6,  # Multiple thermal shutdowns
        "wifi_signal_dbm": -88,
        "wifi_disconnects": 18,
        "wifi_firmware_crashes": 4,
        "wifi_packet_loss_percent": 12.5,
        "cellular_rsrp_dbm": -108,
        "cellular_drop_calls": 8,
        "cellular_modem_resets": 6,
        "cellular_handshake_failures": 15,
        "peripheral_touchscreen_errors": 5,
        "peripheral_camera_init_failures": 8,
        "peripheral_mic_speaker_faults": 4,
        "peripheral_sensor_hub_desyncs": 12,
        "kernel_panics": 18,  # EXCESSIVE KERNEL PANICS
        "kernel_anrs": 35,
        "kernel_irq_storms": 12,
        "kernel_driver_probe_failures": 15,
        "kernel_system_crash_count": 25,
        "kernel_last_boot_reason": "kernel_panic",
        "user_region": "EU",
        "user_account_age_days": 2890,
        "user_device_count": 6,
        "user_failure_history_count": 4,  # Multiple device failures
        "user_behavior_profile": "power_user",
        "failure_occurred": True,
        "failure_type": "system",
        "failure_timestamp": "2025-08-19T18:55:00"
    }

    print("Example Usage:")
    print("="*50)
    

    # Construct full paths
    MODEL_PATH = os.path.join(BASE_MODEL_DIR, "model.joblib")
    PREPROCESSING_PATH = os.path.join(BASE_MODEL_DIR, "preprocessing_pipeline.joblib")

    
    # Load the trained model
    try:
        predictor = ProductionPredictor(
            MODEL_PATH, 
            PREPROCESSING_PATH
        )
        
        # Make prediction
        prediction = predictor.predict_failure(system_failure_telemetry)
        
        print("Prediction Result:")
        print(json.dumps(prediction, indent=2))
        
        # Initialize monitor
        monitor = ModelMonitor()
        monitor.log_prediction(prediction)
        
        print("\nMonitoring Report:")
        report = monitor.generate_performance_report()
        print(json.dumps(report, indent=2))
        
    except FileNotFoundError:
        print("Model files not found. Please run the training pipeline first.")


if __name__ == '__main__':
    example_usage()