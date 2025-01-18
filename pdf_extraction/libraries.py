import os
import uuid
import pandas as pd

from langchain_openai import ChatOpenAI,AzureChatOpenAI
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import Dict, List, TypedDict, Any
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic_core import CoreSchema

from pydantic import BaseModel, GetJsonSchemaHandler

import re
import json
import time

from langchain_community.callbacks import get_openai_callback


from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser

from langchain_core.runnables import RunnableParallel

import timeit


class ProductLineItems(BaseModel):
    """Information about product line items."""
    
    uom: Optional[str] = Field(..., description="Extract the primary unit of measurement for the product line item. The unit of measurement can be in various forms such as each, roll, rolls, box, boxes, pack, packs, piece, set, kit, bundle, pk. If there are multiple UoMs, extract the most relevant to the line. Exclusion: Strictly avoid any unit containing 'inch','in'. If not found, return an empty string.",
    )
    currency: Optional[str] = Field(..., description="Extract the currency of the product for the product line item. The currency should be represented as a string, if no currency is mentioned, infer from the email, if possible. If not found, return an empty string.",
    )    
    quantity: Optional[int] = Field(..., description="Extract the quantity of the product for the product line item. Convert alphabetical numbers to numerical values for ease of processing. Sometimes quantity can also be found separated by '-'(eg: 6'-12/3 so where 6 is the quantity, and 'ft' is the uom). Avoid extracting line number as a quantity. If not found, return 1.",
    )   
    unit_price : Optional[float] = Field(..., description= "Extract the unit price of the product for the product line item. The unit price should be represented as a number, excluding the currency symbol. If not found, return an empty string.",
    )           
    part_number : Optional[str] = Field(..., description= "Extract the part number for the product line item. The part number could be labeled as part number, product number, WTC, material number, PART, MPN #, item Number, item id, manufacturer part number, item #, model number, model #, Vendor Material No., Your material number, type number and mfg Part Number. Once identified if there are multiple labels include all these separating them with a comma, don't include the labels ('part number:', etc.), only have the values. Exclude details such as PO numbers, job numbers, serial number, price, unit of measurement, uom, quantity, and product description. Exclude subject line.",
    )             
    product_description : Optional[str] = Field(..., description= "Extract the product description for the product line item. The product description should include the product name, product description and attributes (size, colour, material, type, length, Type Designation, MNFG, Dimension, Operating voltage and other product attributes such as Height, Width, Voltage type etc.), include all these, separating them with a comma if there are multiple. There may be different colours present in a single line item, Extract them as separate line items. There can be different length present in a single line item, Extract them as separate lines. Exclude details such as purchase order (PO) numbers, serial number, price, unit of measurement (UoM), quantity, part number, product number, WTC, material number, PART, MPN #, item number, item id, manufacturer part number, Delivery Date, item #, vendor material number, shipping instructions, comments, notes, Account number, 'Tag' and Release Detail. Strictly do not include subject line into consideration.",
    )        
    spell_corrected_product_description : Optional[str] = Field(..., description= """Correct any spelling errors, common typing errors, abbreviations, and product-specific jargons in the extracted product details. Use the context of the email to determine the correct spelling. This field is to ensure the accuracy and readability of the product details. Convert any vulgar fractions to normal fractions. Do not return in Unicode characters. Convert double quote " as in (eg: 6/32” to 6/32 in) and single quote ' as ft(eg: 1/2' to 1/2 ft).""",
    )       
    query_wout_uom : Optional[str] = Field(..., description= "Extract Product query and description without any Unit of measure only cardinals and attributes should be present. example : 2  inch pvc conduit or 2' pvc conduit will become output: 2 pvc conduit. Note : Dont change sequence of words in query , preseve the same order in the final query",
    )  
    product_metadata : Optional[str] = Field(..., description= "If the product description is available then try to breakdown the description based on entities and form a metadata in given format, for example: 2 pvc conduit output: productname:conduit|material:pvc|size:2, always try to extract the product name at minimum, USE Your knowledge base to determine the enteties precisely, Also try your best to generate product name in the scenario where you are confident, Dont add quantity in attributes.",
    )   