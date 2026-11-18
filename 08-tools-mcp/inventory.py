"""A read-only teaching tool. Quantities are fictional, not model knowledge."""

from mcp.server.fastmcp import FastMCP

server = FastMCP("classroom-inventory")
STOCK = {"P28": 37, "A17": 0}


@server.tool()
def get_stock(sku: str) -> str:
    """Look up current stock by product code (SKU). Unknown does not mean zero."""
    sku = sku.strip().upper()
    return f"{sku}：{STOCK[sku]} 件" if sku in STOCK else f"{sku}：没有这条记录"


if __name__ == "__main__":
    server.run(transport="stdio")
