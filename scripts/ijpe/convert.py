"""Cache the sheets of the WMS export as parquet (needs openpyxl):

    uv run --with openpyxl python -m scripts.ijpe.convert
"""

from scripts.ijpe.wms_export import convert
from scripts.ijpe import CACHE, EXPORT

if __name__ == "__main__":
    convert(EXPORT, CACHE)
    print(f"cached {EXPORT.name} in {CACHE}")
