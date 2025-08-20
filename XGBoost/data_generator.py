"""
Production-grade ML Pipeline for Device Failure Prediction
Memory-efficient version that processes data in batches to handle billions of records.
"""

import logging
from typing import Dict, Generator
from datetime import datetime, timedelta
import warnings
import csv
import os
warnings.filterwarnings('ignore')

# Core data processing
import numpy as np

# Utilities
from tqdm import tqdm

BASE_DIR = "data"
DATA_FILE_NAME = "data.csv"
NUMBER_OF_RECORDS = 1_000_000
BATCH_SIZE = 50_000

def generate_record_batch(batch_size: int, start_index: int = 0) -> Generator[Dict, None, None]:
    """Generate a batch of telemetry records without storing them all in memory"""
    
    # Device models and manufacturers
    device_models = ["Samsung-Galaxy-S22", "iPhone-14", "Pixel-7", "OnePlus-10", "Xiaomi-12"]
    manufacturers = ["Samsung", "Apple", "Google", "OnePlus", "Xiaomi"]
    
    for i in range(start_index, start_index + batch_size):
        # Base device info
        device_model = np.random.choice(device_models)
        manufacturer = manufacturers[device_models.index(device_model)]
        
        # Generate realistic telemetry first
        battery_temp = np.random.normal(35, 5)
        cpu_temp = np.random.normal(45, 8)
        
        # Create base record with all telemetry
        record = {
            # Identifiers
            "record_id": f"rec_{i:08d}",
            "device_id": f"dev_{i//10:06d}",  # Multiple records per device
            "user_id": f"user_{i//50:05d}",   # Multiple devices per user
            "timestamp": (datetime.now() - timedelta(days=np.random.randint(0, 365))).isoformat(),
            
            # Device metadata
            "device_model": device_model,
            "device_manufacturer": manufacturer,
            "device_batch_number": f"batch_{np.random.randint(1, 100):03d}",
            "device_production_date": (datetime.now() - timedelta(days=np.random.randint(30, 730))).strftime("%Y-%m-%d"),
            "device_age_days": np.random.randint(30, 730),
            "warranty_status": np.random.choice(["in_warranty", "expired"], p=[0.7, 0.3]),
            
            # OS & Software
            "os_name": "Android" if manufacturer != "Apple" else "iOS",
            "os_version": str(np.random.randint(11, 15)),
            "os_build_number": f"build_{np.random.randint(1000, 9999)}",
            "software_firmware_version": f"fw_{np.random.randint(1, 10)}.{np.random.randint(0, 99)}",
            "software_security_patch": (datetime.now() - timedelta(days=np.random.randint(0, 90))).strftime("%Y-%m-%d"),
            "software_uptime_hours": np.random.exponential(24),
            "software_system_crashes": np.random.poisson(1),
            "software_app_crashes": np.random.poisson(2),
            
            # Battery telemetry - Generate realistic values
            "battery_level_percent": np.random.uniform(5, 100),
            "battery_health_status": np.random.choice(["good", "fair", "poor"], p=[0.7, 0.2, 0.1]),
            "battery_temperature_c": battery_temp,
            "battery_voltage_mv": np.random.randint(3500, 4200),
            "battery_charge_cycles": np.random.randint(50, 2000),
            "battery_charging_status": np.random.choice(["charging", "discharging"], p=[0.3, 0.7]),
            "battery_current_ma": np.random.randint(-2000, 2000),
            "battery_resistance_mohm": np.random.randint(100, 500),
            "battery_sudden_shutdowns": np.random.poisson(0.1),
            
            # CPU telemetry
            "cpu_usage_percent": np.random.uniform(10, 90),
            "cpu_temperature_c_avg": cpu_temp,
            "cpu_throttle_events": np.random.poisson(2),
            "cpu_core_frequencies_mhz": ",".join([str(np.random.randint(1000, 3000)) for _ in range(8)]),
            "cpu_voltage_scaling_errors": np.random.poisson(0.5),
            "cpu_watchdog_resets": np.random.poisson(0.1),
            
            # GPU telemetry
            "gpu_usage_percent": np.random.uniform(0, 80),
            "gpu_temperature_c": np.random.normal(40, 8),
            "gpu_driver_resets": np.random.poisson(0.2),
            "gpu_frequency_mhz": np.random.randint(400, 1000),
            
            # Memory telemetry
            "memory_total_mb": np.random.choice([4096, 6144, 8192, 12288]),
            "memory_used_mb": 0,  # Will be calculated
            "memory_swap_used_mb": np.random.randint(0, 1024),
            "memory_page_faults": np.random.poisson(100),
            "memory_oom_kills": np.random.poisson(0.1),
            "memory_ecc_corrected_errors": np.random.poisson(0.05),
            "memory_dma_faults": np.random.poisson(0.02),
            
            # Storage telemetry
            "storage_total_gb": np.random.choice([64, 128, 256, 512, 1024]),
            "storage_used_gb": 0,  # Will be calculated
            "storage_read_errors": np.random.poisson(0.5),
            "storage_write_errors": np.random.poisson(0.3),
            "storage_bad_block_count": np.random.poisson(2),
            "storage_io_latency_ms": np.random.exponential(10),
            "storage_wear_level_percent": np.random.uniform(0, 50),
            
            # Thermal sensors
            "thermal_hotspot_temp_c": np.random.normal(50, 10),
            "thermal_sensor_battery_c": battery_temp + np.random.normal(0, 2),
            "thermal_sensor_cpu_cluster0_c": cpu_temp + np.random.normal(0, 3),
            "thermal_sensor_cpu_cluster1_c": cpu_temp + np.random.normal(2, 3),
            "thermal_sensor_gpu_c": np.random.normal(42, 8),
            "thermal_sensor_pmic_c": np.random.normal(38, 5),
            "thermal_shutdowns": np.random.poisson(0.1),
            
            # Connectivity
            "wifi_signal_dbm": np.random.randint(-80, -30),
            "wifi_disconnects": np.random.poisson(1),
            "wifi_firmware_crashes": np.random.poisson(0.1),
            "wifi_packet_loss_percent": np.random.exponential(2),
            "cellular_rsrp_dbm": np.random.randint(-120, -60),
            "cellular_drop_calls": np.random.poisson(0.5),
            "cellular_modem_resets": np.random.poisson(0.2),
            "cellular_handshake_failures": np.random.poisson(1),
            
            # Peripheral sensors
            "peripheral_touchscreen_errors": np.random.poisson(0.3),
            "peripheral_camera_init_failures": np.random.poisson(0.2),
            "peripheral_mic_speaker_faults": np.random.poisson(0.1),
            "peripheral_sensor_hub_desyncs": np.random.poisson(0.5),
            
            # Kernel & System
            "kernel_panics": np.random.poisson(0.1),
            "kernel_anrs": np.random.poisson(2),
            "kernel_irq_storms": np.random.poisson(0.05),
            "kernel_driver_probe_failures": np.random.poisson(0.3),
            "kernel_system_crash_count": np.random.poisson(0.5),
            "kernel_last_boot_reason": np.random.choice(["normal", "crash", "watchdog", "thermal"]),
            
            # User metadata
            "user_region": np.random.choice(["US", "EU", "ASIA", "OTHER"]),
            "user_account_age_days": np.random.randint(30, 2000),
            "user_device_count": np.random.randint(1, 5),
            "user_failure_history_count": np.random.poisson(1),
            "user_behavior_profile": np.random.choice(["light_user", "normal_user", "heavy_gamer", "business_user"]),
        }
        
        # Calculate dependent fields
        record["memory_used_mb"] = int(record["memory_total_mb"] * np.random.uniform(0.3, 0.9))
        record["storage_used_gb"] = int(record["storage_total_gb"] * np.random.uniform(0.1, 0.8))
        
        # Now determine failure type based on telemetry conditions
        failure_type = determine_failure_type_from_telemetry(record)
        failure_occurred = 1 if failure_type != 'none' else 0
        
        # Add failure information
        record.update({
            "failure_occurred": failure_occurred,
            "failure_type": failure_type,
            "failure_timestamp": (datetime.now() - timedelta(hours=np.random.randint(0, 24))).isoformat() if failure_occurred else ""
        })
        
        yield record


def generate_sample_data(num_records: int = 10_000_000, batch_size: int = 50_000, output_path: str = f'{BASE_DIR}/{DATA_FILE_NAME}') -> None:
    """Generate sample telemetry data efficiently using batched processing"""
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Generating {num_records:,} sample records in batches of {batch_size:,}")
    
    np.random.seed(42)
    
    # Write header first
    sample_record = next(generate_record_batch(1, 0))
    fieldnames = list(sample_record.keys())
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process in batches
        processed_records = 0
        with tqdm(total=num_records, desc="Generating records") as pbar:
            while processed_records < num_records:
                current_batch_size = min(batch_size, num_records - processed_records)
                
                # Generate and write batch
                batch_records = list(generate_record_batch(current_batch_size, processed_records))
                writer.writerows(batch_records)
                
                processed_records += current_batch_size
                pbar.update(current_batch_size)
                
                # Clear batch from memory
                del batch_records
    
    logging.info(f"Sample data saved to {output_path}")
    logging.info(f"File size: {os.path.getsize(output_path) / (1024**3):.2f} GB")


def determine_failure_type_from_telemetry(record: Dict) -> str:
    """Determine realistic failure type based on telemetry conditions"""
    failure_indicators = {
        'battery': [],
        'thermal': [],
        'cpu': [],
        'storage': [],
        'connectivity': [],
        'memory': []
    }
    
    # Battery failure indicators
    if record.get('battery_temperature_c', 0) > 50:
        failure_indicators['battery'].append('high_temp')
    if record.get('battery_level_percent', 100) < 10:
        failure_indicators['battery'].append('low_level')
    if record.get('battery_charge_cycles', 0) > 1000:
        failure_indicators['battery'].append('high_cycles')
    if record.get('battery_sudden_shutdowns', 0) > 0:
        failure_indicators['battery'].append('sudden_shutdown')
    
    # Thermal failure indicators
    if record.get('cpu_temperature_c_avg', 0) > 70:
        failure_indicators['thermal'].append('cpu_overheat')
    if record.get('thermal_hotspot_temp_c', 0) > 65:
        failure_indicators['thermal'].append('hotspot_overheat')
    if record.get('thermal_shutdowns', 0) > 0:
        failure_indicators['thermal'].append('thermal_shutdown')
    
    # CPU failure indicators
    if record.get('cpu_throttle_events', 0) > 10:
        failure_indicators['cpu'].append('throttling')
    if record.get('cpu_watchdog_resets', 0) > 0:
        failure_indicators['cpu'].append('watchdog_reset')
    if record.get('cpu_voltage_scaling_errors', 0) > 2:
        failure_indicators['cpu'].append('voltage_error')
    
    # Storage failure indicators
    if record.get('storage_read_errors', 0) > 5:
        failure_indicators['storage'].append('read_errors')
    if record.get('storage_write_errors', 0) > 3:
        failure_indicators['storage'].append('write_errors')
    if record.get('storage_bad_block_count', 0) > 10:
        failure_indicators['storage'].append('bad_blocks')
    if record.get('storage_wear_level_percent', 0) > 80:
        failure_indicators['storage'].append('high_wear')
    
    # Connectivity failure indicators
    if record.get('wifi_disconnects', 0) > 5:
        failure_indicators['connectivity'].append('wifi_issues')
    if record.get('cellular_drop_calls', 0) > 3:
        failure_indicators['connectivity'].append('cellular_issues')
    if record.get('wifi_firmware_crashes', 0) > 0:
        failure_indicators['connectivity'].append('wifi_crashes')
    
    # Memory failure indicators
    if record.get('memory_oom_kills', 0) > 0:
        failure_indicators['memory'].append('oom_kills')
    if record.get('memory_ecc_corrected_errors', 0) > 0:
        failure_indicators['memory'].append('ecc_errors')
    if record.get('memory_dma_faults', 0) > 0:
        failure_indicators['memory'].append('dma_faults')
    
    # Calculate failure scores
    failure_scores = {}
    for failure_type, indicators in failure_indicators.items():
        failure_scores[failure_type] = len(indicators)
    
    # Determine most likely failure type
    max_score = max(failure_scores.values())
    if max_score >= 2:  # At least 2 indicators for a failure
        # Get failure type with highest score
        for failure_type, score in failure_scores.items():
            if score == max_score:
                return failure_type
    
    return 'none'


if __name__ == "__main__":
    generate_sample_data(num_records=NUMBER_OF_RECORDS, batch_size=BATCH_SIZE)