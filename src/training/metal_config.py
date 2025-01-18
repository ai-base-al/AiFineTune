# File: src/training/metal_config.py

import mlx.core as mx
import psutil

def get_available_memory():
    """Get available system memory in GB"""
    memory = psutil.virtual_memory()
    return memory.available / (1024 * 1024 * 1024)

def init_metal():
    try:
        if not mx.metal.is_available():
            print("Metal is not available on this system")
            return None, False
            
        device = mx.Device(mx.DeviceType.gpu, 0)
        mx.set_default_device(device)
        
        test_tensor = mx.ones((2, 2))
        mx.eval(test_tensor)
        print("Metal test successful")
        
        info = mx.metal.device_info()
        print(f"\nMetal Configuration:")
        print(f"• Architecture: {info['architecture']}")
        print(f"• Memory Size: {info['memory_size'] / (1024**3):.1f}GB")
        print(f"• Max Buffer: {info['max_buffer_length'] / (1024**3):.1f}GB")
        
        # Convert to integer bytes
        max_mem = int(info['memory_size'] * 0.8)  # Use 80% of available memory
        mx.metal.set_memory_limit(max_mem)
        mx.metal.set_cache_limit(max_mem // 4)
        
        return device, True
        
    except Exception as e:
        print(f"Metal initialization failed: {e}")
        device = mx.Device(mx.DeviceType.cpu, 0)
        mx.set_default_device(device)
        return device, False

def configure_metal(device):
    """Basic Metal configuration"""
    if device.type == mx.DeviceType.gpu:
        try:
            mx.metal.clear_cache()
            mx.metal.reset_peak_memory()
            return True
        except Exception as e:
            print(f"Metal configuration warning: {e}")
            return True
    return False

def monitor_memory(step="Unknown"):
    """Monitor memory usage"""
    try:
        if mx.metal.is_available():
            active_mem = mx.metal.get_active_memory()
            peak_mem = mx.metal.get_peak_memory()
            cache_mem = mx.metal.get_cache_memory()
            
            # Convert to GB
            active_gb = active_mem / (1024 * 1024 * 1024)
            peak_gb = peak_mem / (1024 * 1024 * 1024)
            cache_gb = cache_mem / (1024 * 1024 * 1024)
            
            print(f"\nMemory at {step}:")
            print(f"  Active: {active_gb:.2f}GB")
            print(f"  Peak: {peak_gb:.2f}GB")
            print(f"  Cache: {cache_gb:.2f}GB")
            
            if active_gb > peak_gb * 0.8:
                print("⚠️ High memory usage detected!")
                clear_memory()
                
            return {
                "active_gb": active_gb,
                "peak_gb": peak_gb,
                "cache_gb": cache_gb
            }
    except Exception as e:
        print(f"Could not monitor memory: {e}")
        return None

def clear_memory():
    """Clean up memory"""
    if mx.metal.is_available():
        try:
            mx.metal.clear_cache()
            mx.metal.reset_peak_memory()
            import gc
            gc.collect()
        except Exception as e:
            print(f"Error during memory cleanup: {e}")