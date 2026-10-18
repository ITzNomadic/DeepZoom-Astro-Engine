"""
Custom exceptions for DeepZoom Studio.
"""

class DeepZoomError(Exception):
    """Base exception for all DeepZoom Studio errors."""
    pass

class HardwareDetectionError(DeepZoomError):
    """Raised when hardware detection (Vulkan/GPU) fails or behaves unexpectedly."""
    pass

class ModelNotFoundError(DeepZoomError):
    """Raised when a specified model cannot be found in the models directory."""
    pass

class InferenceError(DeepZoomError):
    """Raised when the super-resolution inference engine returns a non-zero exit code or fails."""
    pass

class ValidationError(DeepZoomError):
    """Raised when an output image fails integrity or metric validation."""
    pass
