# Logging Configuration Files

This directory contains all logging configuration files for OntoCast.

## File Organization

### Basic Configurations (Non-Colored)
- `logging.debug.conf` - Standard debug level logging
- `logging.info.conf` - Standard info level logging  
- `logging.warning.conf` - Standard warning level logging

### Rich Colored Configurations (Recommended)
- `logging.debug.colored.conf` - Rich colored debug logging
- `logging.info.colored.conf` - Rich colored info logging
- `logging.warning.colored.conf` - Rich colored warning logging

### Enhanced Rich Configurations
- `logging.debug.rich.conf` - Rich colored debug with function:line numbers
- `logging.info.rich.conf` - Rich colored info with function names
- `logging.warning.rich.conf` - Rich colored warning with function names

### Simple ANSI Configurations (Lightweight)
- `logging.info.simple.conf` - Simple ANSI colored info logging

## Usage

Use with `--logging-level` parameter (without .conf extension):

```bash
uv run serve --logging-level info.rich [options...]
```

See `../LOGGING_COLORS.md` for complete documentation. 