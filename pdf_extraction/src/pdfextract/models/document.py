from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Header(BaseModel):
    """Document header information."""

    purchase_order: Optional[str] = Field(default=None, description="Purchase order number (PO#)")
    invoice_order: Optional[str] = Field(default=None, description="Invoice number")
    ship_to: Optional[str] = Field(default=None, description="Shipping address")
    bill_to: Optional[str] = Field(default=None, description="Billing address")
    vendor: Optional[str] = Field(default=None, description="Vendor / supplier name and details")


class ProductLineItem(BaseModel):
    """A single product line item extracted from an invoice or purchase order."""

    uom: Optional[str] = Field(
        default=None,
        description=(
            "Unit of measurement (each, roll, box, pack, piece, set, kit, bundle). "
            "Exclude anything containing 'inch' or 'in'."
        ),
    )
    currency: Optional[str] = Field(
        default=None,
        description="Currency code or symbol (e.g. USD, EUR, $). Infer from context if not explicit.",
    )
    quantity: Optional[int] = Field(
        default=1,
        description="Quantity of the product. Convert written numbers to integers. Default 1 if not found.",
    )
    unit_price: Optional[float] = Field(
        default=None,
        description="Unit price excluding currency symbol.",
    )
    part_number: Optional[str] = Field(
        default=None,
        description=(
            "Part number — may be labelled as part #, product #, MPN, item #, model #, "
            "WTC, material number, manufacturer part number. Include all values found, comma-separated."
        ),
    )
    product_description: Optional[str] = Field(
        default=None,
        description=(
            "Full product description including name, size, colour, material, type, length, "
            "voltage, and other attributes. Separate multiple attributes with commas. "
            "Exclude PO numbers, serial numbers, price, UoM, quantity, and shipping instructions."
        ),
    )
    spell_corrected_product_description: Optional[str] = Field(
        default=None,
        description=(
            "Spell-corrected version of product_description. Fix abbreviations, typos, and jargon. "
            'Convert vulgar fractions. Replace " with "in" and \' with "ft".'
        ),
    )
    query_wout_uom: Optional[str] = Field(
        default=None,
        description=(
            "Product description without unit of measure — only cardinal numbers and attributes. "
            "Example: '2 inch pvc conduit' → '2 pvc conduit'. Preserve word order."
        ),
    )
    product_metadata: Optional[str] = Field(
        default=None,
        description=(
            "Structured metadata as pipe-separated key:value pairs. "
            "Example: 'productname:conduit|material:pvc|size:2'. "
            "Always include productname at minimum. Do not include quantity."
        ),
    )


class ExtractionResult(BaseModel):
    """Complete extraction result from a PDF document."""

    header: Optional[Header] = None
    line_items: List[ProductLineItem] = Field(default_factory=list)


class ExtractionResponse(BaseModel):
    """API response for a PDF extraction job."""

    job_id: str
    status: str  # pending | processing | completed | failed
    filename: str
    error: Optional[str] = None
    result: Optional[ExtractionResult] = None
