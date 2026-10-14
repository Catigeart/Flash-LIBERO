"""Versioned shared metadata, binding and device admission protocol.

Mathematical producers remain owned by their D/C/S/N modules.
"""
from .schema import FieldDescriptor, checked_product, content_sha256
from .binding import BindingRegistry, BindingError
