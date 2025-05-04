from .illegal_content_scanner import IllegalContentScanner

NODE_CLASS_MAPPINGS = {
    "IllegalContentScanner": IllegalContentScanner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IllegalContentScanner": "🛡️ Illegal Content Scanner"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
