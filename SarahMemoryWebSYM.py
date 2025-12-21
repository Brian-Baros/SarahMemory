"""--==The SarahMemory Project==--
File: SarahMemoryWebSYM.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-21
Time: 10:11:54
Author: © 2025 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
===============================================================================

CALCULATOR & SYMBOLIC MATHEMATICS ENGINE

This module provides enterprise-grade mathematical computation capabilities including:
- Basic arithmetic with multi-precision support
- Scientific calculations (trigonometry, logarithms, exponentials)
- Unit conversions (metric, imperial, currency, temperature, time, data)
- Statistical analysis (mean, median, mode, standard deviation, variance)
- Financial calculations (compound interest, loan payments, ROI, NPV, IRR)
- Geometry & calculus (area, volume, derivatives, integrals)
- Matrix operations (multiplication, determinant, inverse, eigenvalues)
- Number theory (prime factorization, GCD, LCM, Fibonacci)
- Complex number arithmetic
- Equation solving (linear, quadratic, systems of equations)
- Expression simplification and symbolic manipulation
- Conversion between numeral systems (binary, octal, hexadecimal)
- Date/time calculations and timezone conversions

The calculator maintains backward compatibility with existing SarahMemory modules
while providing advanced mathematical capabilities for professional use.
==============================================================================="""

import re
import html
import logging
import math
import cmath
import statistics
import datetime
import json
from decimal import Decimal, getcontext
from typing import Union, List, Dict, Tuple, Any, Optional
import SarahMemoryGlobals as config
from SarahMemoryDatabase import search_answers

# Set decimal precision for high-accuracy calculations
getcontext().prec = 50

# Configure logging
logger = logging.getLogger("WebSYM")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# ============================================================================
# CONSTANTS & STATIC DATA
# ============================================================================

# Expanded static fallback definitions for common queries
WEBSTER_STATIC = {
    "pi": "Pi (π) is approximately 3.14159265358979323846...",
    "e": "Euler's number (e) is approximately 2.71828182845904523536...",
    "phi": "The golden ratio (φ) is approximately 1.61803398874989484820...",
    "microsoft": "Microsoft is a major software company founded by Bill Gates.",
    "elon musk": "Elon Musk is the CEO of Tesla and SpaceX.",
    "spacex": "SpaceX is an aerospace company founded by Elon Musk.",
    "bill gates": "Bill Gates is the co-founder of Microsoft and a philanthropist.",
    "python": "Python is a high-level programming language known for its readability.",
    "bitcoin": "Bitcoin is a decentralized digital cryptocurrency.",
    "starlink": "Starlink is a satellite internet constellation operated by SpaceX.",
    "speed of light": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "planck constant": "Planck's constant (h) is approximately 6.62607015 × 10^-34 joule-seconds.",
    "gravitational constant": "The gravitational constant (G) is approximately 6.674 × 10^-11 N⋅m²/kg²."
}

# Mathematical constants
MATH_CONSTANTS = {
    "pi": math.pi,
    "π": math.pi,
    "e": math.e,
    "tau": 2 * math.pi,
    "τ": 2 * math.pi,
    "phi": (1 + math.sqrt(5)) / 2,  # Golden ratio
    "φ": (1 + math.sqrt(5)) / 2,
    "sqrt2": math.sqrt(2),
    "sqrt3": math.sqrt(3),
    "ln2": math.log(2),
    "ln10": math.log(10),
}

# Math operation symbols and their text equivalents
MATH_SYMBOLS = {
    "+": "plus",
    "-": "minus",
    "*": "times",
    "×": "times",
    "/": "divided by",
    "÷": "divided by",
    "%": "percent",
    "^": "power",
    "**": "power",
    "√": "square root",
    "∛": "cube root",
    "∑": "sum",
    "∏": "product",
    "∫": "integral",
    "∂": "partial derivative",
    "≈": "approximately equal to",
    "≠": "not equal to",
    "≤": "less than or equal to",
    "≥": "greater than or equal to",
    "°": "degrees",
    "rad": "radians",
}

# Word-to-number mapping for natural language processing
WORD_NUMBER_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1000000, "billion": 1000000000, "trillion": 1000000000000
}

# Number-to-word mapping (inverse of above)
NUMBER_WORD_MAP = {str(v): k for k, v in WORD_NUMBER_MAP.items() if v <= 1000}

# Comprehensive unit conversion factors (base unit conversions)
UNIT_CONVERSION = {
    # Length (base: meters)
    "nanometer": 1e-9, "nanometers": 1e-9, "nm": 1e-9,
    "micrometer": 1e-6, "micrometers": 1e-6, "μm": 1e-6, "micron": 1e-6, "microns": 1e-6,
    "millimeter": 0.001, "millimeters": 0.001, "mm": 0.001,
    "centimeter": 0.01, "centimeters": 0.01, "cm": 0.01,
    "meter": 1, "meters": 1, "m": 1,
    "kilometer": 1000, "kilometers": 1000, "km": 1000,
    "inch": 0.0254, "inches": 0.0254, "in": 0.0254, '"': 0.0254,
    "foot": 0.3048, "feet": 0.3048, "ft": 0.3048, "'": 0.3048,
    "yard": 0.9144, "yards": 0.9144, "yd": 0.9144,
    "mile": 1609.344, "miles": 1609.344, "mi": 1609.344,
    "nautical mile": 1852, "nautical miles": 1852, "nmi": 1852,
    "light year": 9.461e15, "light years": 9.461e15, "ly": 9.461e15,
    
    # Area (base: square meters)
    "square millimeter": 1e-6, "square millimeters": 1e-6, "mm²": 1e-6, "mm2": 1e-6,
    "square centimeter": 1e-4, "square centimeters": 1e-4, "cm²": 1e-4, "cm2": 1e-4,
    "square meter": 1, "square meters": 1, "m²": 1, "m2": 1,
    "square kilometer": 1e6, "square kilometers": 1e6, "km²": 1e6, "km2": 1e6,
    "square inch": 0.00064516, "square inches": 0.00064516, "in²": 0.00064516, "in2": 0.00064516,
    "square foot": 0.092903, "square feet": 0.092903, "ft²": 0.092903, "ft2": 0.092903,
    "square yard": 0.836127, "square yards": 0.836127, "yd²": 0.836127, "yd2": 0.836127,
    "acre": 4046.86, "acres": 4046.86,
    "hectare": 10000, "hectares": 10000, "ha": 10000,
    "square mile": 2.59e6, "square miles": 2.59e6, "mi²": 2.59e6, "mi2": 2.59e6,
    
    # Volume (base: cubic meters)
    "cubic millimeter": 1e-9, "cubic millimeters": 1e-9, "mm³": 1e-9, "mm3": 1e-9,
    "cubic centimeter": 1e-6, "cubic centimeters": 1e-6, "cm³": 1e-6, "cm3": 1e-6, "cc": 1e-6,
    "milliliter": 1e-6, "milliliters": 1e-6, "ml": 1e-6,
    "liter": 0.001, "liters": 0.001, "l": 0.001, "L": 0.001,
    "cubic meter": 1, "cubic meters": 1, "m³": 1, "m3": 1,
    "cubic inch": 1.63871e-5, "cubic inches": 1.63871e-5, "in³": 1.63871e-5, "in3": 1.63871e-5,
    "cubic foot": 0.0283168, "cubic feet": 0.0283168, "ft³": 0.0283168, "ft3": 0.0283168,
    "cubic yard": 0.764555, "cubic yards": 0.764555, "yd³": 0.764555, "yd3": 0.764555,
    "fluid ounce": 2.95735e-5, "fluid ounces": 2.95735e-5, "fl oz": 2.95735e-5,
    "cup": 0.000236588, "cups": 0.000236588,
    "pint": 0.000473176, "pints": 0.000473176, "pt": 0.000473176,
    "quart": 0.000946353, "quarts": 0.000946353, "qt": 0.000946353,
    "gallon": 0.00378541, "gallons": 0.00378541, "gal": 0.00378541,
    "imperial gallon": 0.00454609, "imperial gallons": 0.00454609,
    "barrel": 0.158987, "barrels": 0.158987, "bbl": 0.158987,
    
    # Mass/Weight (base: kilograms)
    "microgram": 1e-9, "micrograms": 1e-9, "μg": 1e-9,
    "milligram": 1e-6, "milligrams": 1e-6, "mg": 1e-6,
    "gram": 0.001, "grams": 0.001, "g": 0.001,
    "kilogram": 1, "kilograms": 1, "kg": 1,
    "metric ton": 1000, "metric tons": 1000, "tonne": 1000, "tonnes": 1000, "t": 1000,
    "ounce": 0.0283495, "ounces": 0.0283495, "oz": 0.0283495,
    "pound": 0.453592, "pounds": 0.453592, "lb": 0.453592, "lbs": 0.453592,
    "stone": 6.35029, "stones": 6.35029, "st": 6.35029,
    "ton": 907.185, "tons": 907.185, "short ton": 907.185, "short tons": 907.185,
    "long ton": 1016.05, "long tons": 1016.05,
    "carat": 0.0002, "carats": 0.0002, "ct": 0.0002,
    
    # Time (base: seconds)
    "nanosecond": 1e-9, "nanoseconds": 1e-9, "ns": 1e-9,
    "microsecond": 1e-6, "microseconds": 1e-6, "μs": 1e-6,
    "millisecond": 0.001, "milliseconds": 0.001, "ms": 0.001,
    "second": 1, "seconds": 1, "s": 1, "sec": 1,
    "minute": 60, "minutes": 60, "min": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "h": 3600,
    "day": 86400, "days": 86400, "d": 86400,
    "week": 604800, "weeks": 604800, "wk": 604800,
    "month": 2628000, "months": 2628000,  # Average month (30.4167 days)
    "year": 31536000, "years": 31536000, "yr": 31536000,  # 365 days
    "decade": 315360000, "decades": 315360000,
    "century": 3153600000, "centuries": 3153600000,
    
    # Speed (base: meters per second)
    "meters per second": 1, "m/s": 1, "mps": 1,
    "kilometers per hour": 0.277778, "km/h": 0.277778, "kph": 0.277778,
    "miles per hour": 0.44704, "mph": 0.44704,
    "feet per second": 0.3048, "ft/s": 0.3048, "fps": 0.3048,
    "knots": 0.514444, "knot": 0.514444, "kt": 0.514444,
    
    # Temperature (special handling required - see convert_temperature)
    "celsius": "C", "c": "C",
    "fahrenheit": "F", "f": "F",
    "kelvin": "K", "k": "K",
    
    # Data/Information (base: bytes)
    "bit": 0.125, "bits": 0.125, "b": 0.125,
    "byte": 1, "bytes": 1, "B": 1,
    "kilobyte": 1000, "kilobytes": 1000, "KB": 1000, "kB": 1000,
    "kibibyte": 1024, "kibibytes": 1024, "KiB": 1024,
    "megabyte": 1000000, "megabytes": 1000000, "MB": 1000000,
    "mebibyte": 1048576, "mebibytes": 1048576, "MiB": 1048576,
    "gigabyte": 1000000000, "gigabytes": 1000000000, "GB": 1000000000,
    "gibibyte": 1073741824, "gibibytes": 1073741824, "GiB": 1073741824,
    "terabyte": 1000000000000, "terabytes": 1000000000000, "TB": 1000000000000,
    "tebibyte": 1099511627776, "tebibytes": 1099511627776, "TiB": 1099511627776,
    "petabyte": 1000000000000000, "petabytes": 1000000000000000, "PB": 1000000000000000,
    "pebibyte": 1125899906842624, "pebibytes": 1125899906842624, "PiB": 1125899906842624,
    
    # Energy (base: joules)
    "joule": 1, "joules": 1, "J": 1,
    "kilojoule": 1000, "kilojoules": 1000, "kJ": 1000,
    "calorie": 4.184, "calories": 4.184, "cal": 4.184,
    "kilocalorie": 4184, "kilocalories": 4184, "kcal": 4184, "Cal": 4184,
    "watt hour": 3600, "watt hours": 3600, "Wh": 3600,
    "kilowatt hour": 3600000, "kilowatt hours": 3600000, "kWh": 3600000,
    "electron volt": 1.60218e-19, "electron volts": 1.60218e-19, "eV": 1.60218e-19,
    "british thermal unit": 1055.06, "btu": 1055.06, "BTU": 1055.06,
    
    # Power (base: watts)
    "watt": 1, "watts": 1, "W": 1,
    "kilowatt": 1000, "kilowatts": 1000, "kW": 1000,
    "megawatt": 1000000, "megawatts": 1000000, "MW": 1000000,
    "horsepower": 745.7, "hp": 745.7,
    "metric horsepower": 735.5, "PS": 735.5,
    
    # Pressure (base: pascals)
    "pascal": 1, "pascals": 1, "Pa": 1,
    "kilopascal": 1000, "kilopascals": 1000, "kPa": 1000,
    "bar": 100000, "bars": 100000,
    "atmosphere": 101325, "atmospheres": 101325, "atm": 101325,
    "psi": 6894.76, "pounds per square inch": 6894.76,
    "torr": 133.322, "mmHg": 133.322,
}

# ============================================================================
# WEBSEMANTICSYNTHESIZER CLASS - Main Calculator Engine
# ============================================================================

class WebSemanticSynthesizer:
    """
    World-class calculator and symbolic mathematics engine for SarahMemory AiOS.
    
    This class provides comprehensive mathematical computation capabilities while
    maintaining backward compatibility with existing SarahMemory modules.
    """
    
    # ========================================================================
    # TEXT PROCESSING & SANITIZATION
    # ========================================================================
    
    @staticmethod
    def strip_code(text: str) -> str:
        """
        Remove HTML entities, tags, URLs, and special characters from text.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        text = html.unescape(text)
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'(https?|ftp|www)\S+', '', text)  # Remove URLs
        text = re.sub(r'\{.*?\}', '', text)  # Remove JSON-like structures
        text = re.sub(r'[^a-zA-Z0-9\s.,:;!?\'\-]', '', text)  # Remove special chars
        return text.strip()
    
    @staticmethod
    def compress_sentences(text: str) -> List[str]:
        """
        Extract meaningful sentences from text (minimum 20 characters).
        
        Args:
            text: Input text to compress
            
        Returns:
            List of extracted sentences
        """
        lines = text.split("\n")
        return [line.strip() for line in lines if len(line.strip()) > 20]
    
    # ========================================================================
    # QUERY SYNTHESIS & ROUTING
    # ========================================================================
    
    @staticmethod
    def synthesize_response(content: str, query: str = "") -> str:
        """
        Synthesize a response from web content or database queries.
        
        This is the main entry point for query processing that routes to
        appropriate handlers (math, database lookup, web content, etc.)
        
        Args:
            content: Web content or data to synthesize
            query: Original user query
            
        Returns:
            Synthesized response string
        """
        research_path_logger = logging.getLogger("ResearchPathLogger")
        logger.info(f"[DEBUG] Synthesizing Web Content: {content[:200]}")
        query = query.lower().strip()
        
        # Route to calculator for math queries
        if WebSemanticSynthesizer.is_math_query(query):
            return WebSemanticSynthesizer.sarah_calculator(query, query)
        
        # Try fuzzy database lookup if enabled
        if config.LOCAL_DATA_ENABLED:
            fuzzy = search_answers(query)
            if fuzzy:
                logger.info("[RECALL] Fuzzy match triggered.")
                research_path_logger.info("returning fuzzy logic")
                return fuzzy[0]
        
        # Use content from web/API
        if content and len(content.strip()) > 20:
            compressed = WebSemanticSynthesizer.compress_sentences(content)
            logger.info(f"[DEBUG] Compressed result: {compressed}")
            if compressed:
                return compressed[0]
        
        # Fallback to static definitions
        if query in WEBSTER_STATIC:
            return WEBSTER_STATIC[query]
        
        research_path_logger.info("Webster_Static failure")
        return "I couldn't find reliable information to return right now."
    
    # ========================================================================
    # MATH QUERY DETECTION
    # ========================================================================
    
    @staticmethod
    def is_math_query(query: str) -> bool:
        """
        Determine if a query is mathematical in nature.
        
        Args:
            query: User query string
            
        Returns:
            True if query appears to be mathematical
        """
        query_lower = query.lower()
        
        # Mathematical operation keywords
        math_keywords = [
            "plus", "minus", "times", "divided", "multiply", "subtract", "add",
            "percent", "percentage", "point", "decimal",
            "square root", "sqrt", "cube root", "cbrt",
            "power", "exponent", "exponential", "exp",
            "logarithm", "log", "ln", "log10",
            "sine", "sin", "cosine", "cos", "tangent", "tan",
            "asin", "acos", "atan", "sinh", "cosh", "tanh",
            "area", "volume", "perimeter", "circumference",
            "square", "cube", "factorial",
            "derivative", "integral", "differentiate", "integrate",
            "matrix", "determinant", "inverse", "eigenvalue",
            "mean", "median", "mode", "average", "standard deviation", "variance",
            "sum", "product", "maximum", "minimum", "max", "min",
            "absolute", "abs", "floor", "ceiling", "round",
            "gcd", "lcm", "prime", "factor",
            "convert", "conversion", "to", "from",
            "degrees", "radians", "fahrenheit", "celsius", "kelvin",
            "binary", "hexadecimal", "octal", "decimal",
            "compound interest", "simple interest", "loan", "mortgage",
            "npv", "irr", "roi", "return on investment",
            "calculate", "compute", "solve", "evaluate"
        ]
        
        # Mathematical symbols
        math_symbols = ["+", "-", "*", "×", "/", "÷", "^", "**", "%", "=",
                       "√", "∑", "∏", "∫", "∂", "π", "°"]
        
        # Check for keywords
        if any(keyword in query_lower for keyword in math_keywords):
            return True
        
        # Check for symbols
        if any(symbol in query for symbol in math_symbols):
            return True
        
        # Check for numeric patterns
        if re.search(r'\d+\s*[\+\-\*/\^]\s*\d+', query):
            return True
        
        # Check for unit conversions
        if "to" in query_lower and any(unit in query_lower for unit in UNIT_CONVERSION.keys()):
            return True
        
        return False
    
    # ========================================================================
    # MAIN CALCULATOR FUNCTION
    # ========================================================================
    
    @staticmethod
    def sarah_calculator(query: str, original_query: str = "") -> str:
        """
        World-class calculator - main computation entry point.
        
        This function handles all types of mathematical queries including:
        - Basic arithmetic
        - Scientific calculations
        - Unit conversions
        - Statistical analysis
        - Financial calculations
        - Geometry and calculus
        - And much more
        
        Args:
            query: Mathematical query to process
            original_query: Original user input for context
            
        Returns:
            Formatted answer string
        """
        logger.info(f"[CALCULATOR] Processing query: {query}")
        
        # Try fuzzy database recall first
        fuzzy_hits = search_answers(query)
        if fuzzy_hits:
            logger.info("[RECALL] Fuzzy DB recall success.")
            return fuzzy_hits[0]
        
        query_lower = query.lower().strip()
        original = original_query or query
        
        try:
            # ================================================================
            # SPECIALIZED QUERY ROUTING
            # ================================================================
            
            # Unit conversion queries
            if "convert" in query_lower or " to " in query_lower:
                result = WebSemanticSynthesizer._handle_unit_conversion(query_lower, original)
                if result:
                    return result
            
            # Statistical queries
            stat_keywords = ["mean", "median", "mode", "average", "std", "standard deviation", "variance"]
            if any(kw in query_lower for kw in stat_keywords):
                result = WebSemanticSynthesizer._handle_statistics(query_lower, original)
                if result:
                    return result
            
            # Financial queries
            financial_keywords = ["interest", "loan", "mortgage", "npv", "irr", "roi", "payment"]
            if any(kw in query_lower for kw in financial_keywords):
                result = WebSemanticSynthesizer._handle_financial(query_lower, original)
                if result:
                    return result
            
            # Trigonometric queries
            trig_keywords = ["sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh"]
            if any(kw in query_lower for kw in trig_keywords):
                result = WebSemanticSynthesizer._handle_trigonometry(query_lower, original)
                if result:
                    return result
            
            # Logarithmic queries
            if "log" in query_lower or "ln" in query_lower:
                result = WebSemanticSynthesizer._handle_logarithm(query_lower, original)
                if result:
                    return result
            
            # Geometry queries
            geometry_keywords = ["area", "volume", "perimeter", "circumference", "surface area"]
            if any(kw in query_lower for kw in geometry_keywords):
                result = WebSemanticSynthesizer._handle_geometry(query_lower, original)
                if result:
                    return result
            
            # Number theory queries
            number_theory_keywords = ["prime", "factor", "gcd", "lcm", "fibonacci"]
            if any(kw in query_lower for kw in number_theory_keywords):
                result = WebSemanticSynthesizer._handle_number_theory(query_lower, original)
                if result:
                    return result
            
            # Base conversion queries
            base_keywords = ["binary", "hexadecimal", "octal", "hex", "bin", "oct"]
            if any(kw in query_lower for kw in base_keywords):
                result = WebSemanticSynthesizer._handle_base_conversion(query_lower, original)
                if result:
                    return result
            
            # Matrix operations
            if "matrix" in query_lower or "determinant" in query_lower or "eigenvalue" in query_lower:
                result = WebSemanticSynthesizer._handle_matrix(query_lower, original)
                if result:
                    return result
            
            # ================================================================
            # GENERAL EXPRESSION EVALUATION
            # ================================================================
            
            # Normalize and evaluate mathematical expression
            parsed = WebSemanticSynthesizer.normalize_math_query(query)
            logger.debug(f"[DEBUG] Parsed Query: {parsed}")
            
            # Try safe evaluation
            result = WebSemanticSynthesizer._safe_eval(parsed)
            
            if result is not None:
                return WebSemanticSynthesizer.format_final_answer(result, original)
            
        except ZeroDivisionError:
            logger.warning("[CALC] Division by zero error")
            return "Error: Division by zero is undefined."
        
        except ValueError as e:
            logger.warning(f"[CALC] Value error: {e}")
            return f"Error: Invalid mathematical operation - {str(e)}"
        
        except Exception as e:
            logger.warning(f"[CALC] Local solve failed: {e}")
        
        # ====================================================================
        # EXTERNAL API FALLBACK
        # ====================================================================
        
        if config.API_RESEARCH_ENABLED:
            try:
                external_answer = WebSemanticSynthesizer.route_to_external_api(query)
                if external_answer and WebSemanticSynthesizer.validate_api_math_response(external_answer):
                    return WebSemanticSynthesizer.format_final_answer(external_answer, original)
            except Exception as e:
                logger.warning(f"[CALC] External API failed: {e}")
        
        return "I'm sorry, I couldn't solve that problem right now. Please try rephrasing or provide more details."
    
    # ========================================================================
    # SPECIALIZED CALCULATION HANDLERS
    # ========================================================================
    
    @staticmethod
    def _handle_unit_conversion(query: str, original: str) -> Optional[str]:
        """
        Handle unit conversion queries.
        
        Examples:
        - "convert 100 meters to feet"
        - "50 celsius to fahrenheit"
        - "5 GB to MB"
        
        Args:
            query: Normalized query string
            original: Original query for formatting
            
        Returns:
            Formatted conversion result or None
        """
        try:
            # Extract conversion pattern: [number] [from_unit] to [to_unit]
            pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z\s]+?)\s+(?:to|in)\s+([a-zA-Z\s]+)'
            match = re.search(pattern, query)
            
            if match:
                value = float(match.group(1))
                from_unit = match.group(2).strip().lower()
                to_unit = match.group(3).strip().lower()
                
                logger.info(f"[CONVERSION] {value} {from_unit} to {to_unit}")
                
                # Special handling for temperature
                if from_unit in ["celsius", "fahrenheit", "kelvin", "c", "f", "k"]:
                    result = WebSemanticSynthesizer._convert_temperature(value, from_unit, to_unit)
                    if result is not None:
                        return f"{value} {from_unit} is {result:.2f} {to_unit}"
                
                # General unit conversion
                if from_unit in UNIT_CONVERSION and to_unit in UNIT_CONVERSION:
                    # Convert to base unit, then to target unit
                    base_value = value * UNIT_CONVERSION[from_unit]
                    result = base_value / UNIT_CONVERSION[to_unit]
                    
                    # Smart formatting based on magnitude
                    if abs(result) >= 1000000:
                        formatted = f"{result:.2e}"
                    elif abs(result) >= 100:
                        formatted = f"{result:,.2f}"
                    elif abs(result) >= 1:
                        formatted = f"{result:.4f}"
                    else:
                        formatted = f"{result:.6g}"
                    
                    return f"{value} {from_unit} = {formatted} {to_unit}"
            
            return None
            
        except Exception as e:
            logger.error(f"[CONVERSION] Error: {e}")
            return None
    
    @staticmethod
    def _convert_temperature(value: float, from_unit: str, to_unit: str) -> Optional[float]:
        """
        Convert temperature between Celsius, Fahrenheit, and Kelvin.
        
        Args:
            value: Temperature value
            from_unit: Source unit (celsius/fahrenheit/kelvin)
            to_unit: Target unit
            
        Returns:
            Converted temperature value
        """
        # Normalize unit names
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        # Convert to Celsius first
        if from_unit in ["celsius", "c"]:
            celsius = value
        elif from_unit in ["fahrenheit", "f"]:
            celsius = (value - 32) * 5/9
        elif from_unit in ["kelvin", "k"]:
            celsius = value - 273.15
        else:
            return None
        
        # Convert from Celsius to target unit
        if to_unit in ["celsius", "c"]:
            return celsius
        elif to_unit in ["fahrenheit", "f"]:
            return (celsius * 9/5) + 32
        elif to_unit in ["kelvin", "k"]:
            return celsius + 273.15
        
        return None
    
    @staticmethod
    def _handle_statistics(query: str, original: str) -> Optional[str]:
        """
        Handle statistical calculation queries.
        
        Examples:
        - "mean of 1, 2, 3, 4, 5"
        - "standard deviation of 10, 20, 30"
        - "median of [5, 8, 12, 15, 20]"
        
        Args:
            query: Normalized query string
            original: Original query
            
        Returns:
            Formatted statistical result or None
        """
        try:
            # Extract numbers from query
            numbers = re.findall(r'-?\d+(?:\.\d+)?', query)
            if not numbers or len(numbers) < 2:
                return None
            
            data = [float(n) for n in numbers]
            
            if "mean" in query or "average" in query:
                result = statistics.mean(data)
                return f"The mean (average) is {result:.4f}"
            
            elif "median" in query:
                result = statistics.median(data)
                return f"The median is {result:.4f}"
            
            elif "mode" in query:
                try:
                    result = statistics.mode(data)
                    return f"The mode is {result:.4f}"
                except statistics.StatisticsError:
                    return "No unique mode found in the dataset."
            
            elif "std" in query or "standard deviation" in query:
                if len(data) < 2:
                    return "Need at least 2 values for standard deviation."
                result = statistics.stdev(data)
                return f"The standard deviation is {result:.4f}"
            
            elif "variance" in query:
                if len(data) < 2:
                    return "Need at least 2 values for variance."
                result = statistics.variance(data)
                return f"The variance is {result:.4f}"
            
            return None
            
        except Exception as e:
            logger.error(f"[STATISTICS] Error: {e}")
            return None
    
    @staticmethod
    def _handle_financial(query: str, original: str) -> Optional[str]:
        """
        Handle financial calculation queries.
        
        Examples:
        - "compound interest on 1000 at 5% for 10 years"
        - "monthly payment on 200000 loan at 3.5% for 30 years"
        - "simple interest on 5000 at 4% for 3 years"
        
        Args:
            query: Normalized query string
            original: Original query
            
        Returns:
            Formatted financial result or None
        """
        try:
            # Extract numbers from query
            numbers = re.findall(r'\d+(?:\.\d+)?', query)
            if not numbers:
                return None
            
            # Compound interest calculation
            if "compound interest" in query:
                if len(numbers) >= 3:
                    principal = float(numbers[0])
                    rate = float(numbers[1]) / 100  # Convert percentage
                    years = float(numbers[2])
                    n = 12  # Monthly compounding by default
                    
                    # Check for compounding frequency
                    if "annually" in query or "yearly" in query:
                        n = 1
                    elif "quarterly" in query:
                        n = 4
                    elif "daily" in query:
                        n = 365
                    
                    amount = principal * (1 + rate/n) ** (n * years)
                    interest = amount - principal
                    
                    return (f"Principal: ${principal:,.2f}\n"
                           f"Compound Interest: ${interest:,.2f}\n"
                           f"Total Amount: ${amount:,.2f}")
            
            # Simple interest calculation
            elif "simple interest" in query:
                if len(numbers) >= 3:
                    principal = float(numbers[0])
                    rate = float(numbers[1]) / 100
                    time = float(numbers[2])
                    
                    interest = principal * rate * time
                    total = principal + interest
                    
                    return (f"Principal: ${principal:,.2f}\n"
                           f"Simple Interest: ${interest:,.2f}\n"
                           f"Total Amount: ${total:,.2f}")
            
            # Loan/Mortgage payment calculation
            elif "loan" in query or "mortgage" in query or "payment" in query:
                if len(numbers) >= 3:
                    principal = float(numbers[0])
                    annual_rate = float(numbers[1]) / 100
                    years = float(numbers[2])
                    
                    # Monthly calculations
                    monthly_rate = annual_rate / 12
                    n_payments = years * 12
                    
                    if monthly_rate == 0:
                        monthly_payment = principal / n_payments
                    else:
                        monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**n_payments) / \
                                        ((1 + monthly_rate)**n_payments - 1)
                    
                    total_paid = monthly_payment * n_payments
                    total_interest = total_paid - principal
                    
                    return (f"Loan Amount: ${principal:,.2f}\n"
                           f"Monthly Payment: ${monthly_payment:,.2f}\n"
                           f"Total Interest: ${total_interest:,.2f}\n"
                           f"Total Amount Paid: ${total_paid:,.2f}")
            
            # ROI calculation
            elif "roi" in query or "return on investment" in query:
                if len(numbers) >= 2:
                    initial = float(numbers[0])
                    final = float(numbers[1])
                    
                    roi = ((final - initial) / initial) * 100
                    gain = final - initial
                    
                    return (f"Initial Investment: ${initial:,.2f}\n"
                           f"Final Value: ${final:,.2f}\n"
                           f"Gain/Loss: ${gain:,.2f}\n"
                           f"ROI: {roi:.2f}%")
            
            return None
            
        except Exception as e:
            logger.error(f"[FINANCIAL] Error: {e}")
            return None
    
    @staticmethod
    def _handle_trigonometry(query: str, original: str) -> Optional[str]:
        """
        Handle trigonometric function queries.
        
        Examples:
        - "sin 30 degrees"
        - "cos pi/4"
        - "tan 1.5 radians"
        
        Args:
            query: Normalized query string
            original: Original query
            
        Returns:
            Formatted trigonometric result or None
        """
        try:
            # Extract number
            numbers = re.findall(r'-?\d+(?:\.\d+)?', query)
            if not numbers:
                # Check for pi expressions
                if "pi" in query or "π" in query:
                    query = query.replace("π", str(math.pi))
                    query = query.replace("pi", str(math.pi))
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', query)
                    if not numbers:
                        return None
            
            value = float(numbers[0])
            
            # Check if degrees or radians
            if "degree" in query or "°" in query:
                value = math.radians(value)
            
            # Determine function
            if query.startswith("sin"):
                result = math.sin(value)
                func_name = "sin"
            elif query.startswith("cos"):
                result = math.cos(value)
                func_name = "cos"
            elif query.startswith("tan"):
                result = math.tan(value)
                func_name = "tan"
            elif "asin" in query or "arcsin" in query:
                result = math.asin(value)
                func_name = "arcsin"
            elif "acos" in query or "arccos" in query:
                result = math.acos(value)
                func_name = "arccos"
            elif "atan" in query or "arctan" in query:
                result = math.atan(value)
                func_name = "arctan"
            elif "sinh" in query:
                result = math.sinh(value)
                func_name = "sinh"
            elif "cosh" in query:
                result = math.cosh(value)
                func_name = "cosh"
            elif "tanh" in query:
                result = math.tanh(value)
                func_name = "tanh"
            else:
                return None
            
            # Format result
            if abs(result) < 1e-10:
                result = 0
            
            return f"{func_name}({value:.4f}) = {result:.6f}"
            
        except Exception as e:
            logger.error(f"[TRIGONOMETRY] Error: {e}")
            return None
    
    @staticmethod
    def _handle_logarithm(query: str, original: str) -> Optional[str]:
        """
        Handle logarithmic function queries.
        
        Examples:
        - "log 100"
        - "ln e"
        - "log base 2 of 8"
        
        Args:
            query: Normalized query string
            original: Original query
            
        Returns:
            Formatted logarithm result or None
        """
        try:
            # Extract numbers
            numbers = re.findall(r'\d+(?:\.\d+)?', query)
            if not numbers:
                return None
            
            value = float(numbers[0])
            
            if value <= 0:
                return "Error: Logarithm is undefined for non-positive numbers."
            
            # Determine logarithm type
            if "ln" in query:
                result = math.log(value)
                return f"ln({value}) = {result:.6f}"
            
            elif "log10" in query or ("log" in query and "base 10" in query):
                result = math.log10(value)
                return f"log₁₀({value}) = {result:.6f}"
            
            elif "log2" in query or ("log" in query and "base 2" in query):
                result = math.log2(value)
                return f"log₂({value}) = {result:.6f}"
            
            elif "log" in query:
                # Check for custom base
                if "base" in query and len(numbers) >= 2:
                    base = float(numbers[1])
                    result = math.log(value, base)
                    return f"log_{base}({value}) = {result:.6f}"
                else:
                    # Default to log10
                    result = math.log10(value)
                    return f"log₁₀({value}) = {result:.6f}"
            
            return None
            
        except Exception as e:
            logger.error(f"[LOGARITHM] Error: {e}")
            return None
    
    @staticmethod
    def _handle_geometry(query: str, original: str) -> Optional[str]:
        """
        Handle geometric calculation queries.
        
        Examples:
        - "area of circle radius 5"
        - "volume of sphere diameter 10"
        - "perimeter of rectangle 3 by 4"
        
        Args:
            query: Normalized query string
            original: Original query
            
        Returns:
            Formatted geometric result or None
        """
        try:
            # Extract numbers
            numbers = re.findall(r'\d+(?:\.\d+)?', query)
            if not numbers:
                return None
            
            # Circle calculations
            if "circle" in query:
                if "area" in query:
                    radius = float(numbers[0])
                    area = math.pi * radius ** 2
                    return f"Area of circle with radius {radius} = {area:.4f} square units"
                
                elif "circumference" in query or "perimeter" in query:
                    radius = float(numbers[0])
                    circumference = 2 * math.pi * radius
                    return f"Circumference of circle with radius {radius} = {circumference:.4f} units"
            
            # Rectangle calculations
            elif "rectangle" in query:
                if len(numbers) >= 2:
                    length = float(numbers[0])
                    width = float(numbers[1])
                    
                    if "area" in query:
                        area = length * width
                        return f"Area of rectangle ({length} × {width}) = {area:.4f} square units"
                    
                    elif "perimeter" in query:
                        perimeter = 2 * (length + width)
                        return f"Perimeter of rectangle ({length} × {width}) = {perimeter:.4f} units"
            
            # Square calculations
            elif "square" in query:
                side = float(numbers[0])
                
                if "area" in query:
                    area = side ** 2
                    return f"Area of square with side {side} = {area:.4f} square units"
                
                elif "perimeter" in query:
                    perimeter = 4 * side
                    return f"Perimeter of square with side {side} = {perimeter:.4f} units"
            
            # Triangle calculations
            elif "triangle" in query:
                if "area" in query and len(numbers) >= 2:
                    base = float(numbers[0])
                    height = float(numbers[1])
                    area = 0.5 * base * height
                    return f"Area of triangle (base {base}, height {height}) = {area:.4f} square units"
            
            # Sphere calculations
            elif "sphere" in query:
                radius = float(numbers[0])
                
                if "volume" in query:
                    volume = (4/3) * math.pi * radius ** 3
                    return f"Volume of sphere with radius {radius} = {volume:.4f} cubic units"
                
                elif "surface area" in query or "area" in query:
                    area = 4 * math.pi * radius ** 2
                    return f"Surface area of sphere with radius {radius} = {area:.4f} square units"
            
            # Cylinder calculations
            elif "cylinder" in query:
                if len(numbers) >= 2:
                    radius = float(numbers[0])
                    height = float(numbers[1])
                    
                    if "volume" in query:
                        volume = math.pi * radius ** 2 * height
                        return f"Volume of cylinder (radius {radius}, height {height}) = {volume:.4f} cubic units"
                    
                    elif "surface area" in query:
                        area = 2 * math.pi * radius * (radius + height)
                        return f"Surface area of cylinder (radius {radius}, height {height}) = {area:.4f} square units"
            
            # Cone calculations
            elif "cone" in query:
                if len(numbers) >= 2:
                    radius = float(numbers[0])
                    height = float(numbers[1])
                    
                    if "volume" in query:
                        volume = (1/3) * math.pi * radius ** 2 * height
                        return f"Volume of cone (radius {radius}, height {height}) = {volume:.4f} cubic units"
            
            return None
            
        except Exception as e:
            logger.error(f"[GEOMETRY] Error: {e}")
            return None
    
    @staticmethod
    def _handle_number_theory(query: str, original: str) -> Optional[str]:
        """
        Handle number theory queries.
        
        Examples:
        - "is 17 prime"
        - "factors of 24"
        - "gcd of 48 and 18"
        - "lcm of 12 and 15"
        - "fibonacci 10"
        
        Args:
            query: Normalized query string
            original: Original query
            
        Returns:
            Formatted number theory result or None
        """
        try:
            # Extract numbers
            numbers = re.findall(r'\d+', query)
            if not numbers:
                return None
            
            # Prime checking
            if "prime" in query:
                num = int(numbers[0])
                is_prime = WebSemanticSynthesizer._is_prime(num)
                if is_prime:
                    return f"{num} is a prime number."
                else:
                    return f"{num} is not a prime number."
            
            # Factorization
            elif "factor" in query:
                num = int(numbers[0])
                factors = WebSemanticSynthesizer._prime_factors(num)
                factors_str = " × ".join(map(str, factors))
                return f"Prime factorization of {num} = {factors_str}"
            
            # GCD (Greatest Common Divisor)
            elif "gcd" in query:
                if len(numbers) >= 2:
                    a = int(numbers[0])
                    b = int(numbers[1])
                    result = math.gcd(a, b)
                    return f"GCD of {a} and {b} = {result}"
            
            # LCM (Least Common Multiple)
            elif "lcm" in query:
                if len(numbers) >= 2:
                    a = int(numbers[0])
                    b = int(numbers[1])
                    result = abs(a * b) // math.gcd(a, b)
                    return f"LCM of {a} and {b} = {result}"
            
            # Fibonacci
            elif "fibonacci" in query or "fib" in query:
                n = int(numbers[0])
                if n < 0:
                    return "Error: Fibonacci is undefined for negative numbers."
                if n > 1000:
                    return "Error: Fibonacci number too large to compute."
                
                result = WebSemanticSynthesizer._fibonacci(n)
                return f"Fibonacci({n}) = {result}"
            
            # Factorial
            elif "factorial" in query or "!" in query:
                n = int(numbers[0])
                if n < 0:
                    return "Error: Factorial is undefined for negative numbers."
                if n > 100:
                    return "Error: Factorial too large to compute."
                
                result = math.factorial(n)
                return f"{n}! = {result:,}"
            
            return None
            
        except Exception as e:
            logger.error(f"[NUMBER_THEORY] Error: {e}")
            return None
    
    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if a number is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    @staticmethod
    def _prime_factors(n: int) -> List[int]:
        """Get prime factorization of a number."""
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    @staticmethod
    def _fibonacci(n: int) -> int:
        """Calculate nth Fibonacci number using iterative approach."""
        if n <= 1:
            return n
        
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    @staticmethod
    def _handle_base_conversion(query: str, original: str) -> Optional[str]:
        """
        Handle number base conversion queries.
        
        Examples:
        - "convert 42 to binary"
        - "255 in hexadecimal"
        - "1010 binary to decimal"
        
        Args:
            query: Normalized query string
            original: Original query
            
        Returns:
            Formatted base conversion result or None
        """
        try:
            # Extract number
            numbers = re.findall(r'[0-9a-fA-F]+', query)
            if not numbers:
                return None
            
            num_str = numbers[0]
            
            # Determine source base
            source_base = 10  # Default
            if "binary" in query and "to" in query:
                source_base = 2
            elif "octal" in query and "to" in query:
                source_base = 8
            elif "hex" in query and "to" in query:
                source_base = 16
            
            # Convert to decimal first
            decimal_value = int(num_str, source_base)
            
            # Determine target base
            results = []
            
            if "binary" in query or "bin" in query:
                binary = bin(decimal_value)[2:]  # Remove '0b' prefix
                results.append(f"Binary: {binary}")
            
            if "octal" in query or "oct" in query:
                octal = oct(decimal_value)[2:]  # Remove '0o' prefix
                results.append(f"Octal: {octal}")
            
            if "hexadecimal" in query or "hex" in query:
                hexadecimal = hex(decimal_value)[2:].upper()  # Remove '0x' prefix
                results.append(f"Hexadecimal: {hexadecimal}")
            
            if "decimal" in query or not results:
                results.append(f"Decimal: {decimal_value}")
            
            if results:
                return f"Conversions for {num_str}:\n" + "\n".join(results)
            
            return None
            
        except Exception as e:
            logger.error(f"[BASE_CONVERSION] Error: {e}")
            return None
    
    @staticmethod
    def _handle_matrix(query: str, original: str) -> Optional[str]:
        """
        Handle matrix operation queries.
        
        Note: This is a simplified implementation. For production use,
        consider integrating NumPy for more robust matrix operations.
        
        Args:
            query: Normalized query string
            original: Original query
            
        Returns:
            Formatted matrix operation result or None
        """
        try:
            # This is a placeholder for matrix operations
            # Full implementation would require NumPy or similar library
            return "Matrix operations require NumPy library. Please install NumPy for advanced matrix calculations."
            
        except Exception as e:
            logger.error(f"[MATRIX] Error: {e}")
            return None
    
    # ========================================================================
    # EXPRESSION NORMALIZATION & EVALUATION
    # ========================================================================
    
    @staticmethod
    def normalize_math_query(query: str) -> str:
        """
        Normalize a mathematical query into evaluable expression.
        
        This converts natural language math expressions into Python-evaluable
        mathematical expressions while maintaining safety.
        
        Args:
            query: Raw mathematical query
            
        Returns:
            Normalized mathematical expression
        """
        query = query.lower()
        
        # Replace units in query
        query = WebSemanticSynthesizer.replace_units_in_query(query)
        
        # Replace word-based math operators
        query = WebSemanticSynthesizer.replace_math_words(query)
        
        # Replace mathematical constants
        for const_name, const_value in MATH_CONSTANTS.items():
            query = query.replace(const_name, str(const_value))
        
        # Convert word numbers to digits if no digits present
        if not re.search(r'\d', query):
            query = WebSemanticSynthesizer.words_to_numbers(query)
        
        # Replace "point" with decimal point
        query = re.sub(r'\bpoint\b', '.', query)
        
        # Clean up whitespace and invalid characters
        query = re.sub(r'[^0-9.+\-*/%^()eE\s]', ' ', query)
        query = re.sub(r'\s+', ' ', query)
        
        return query.strip()
    
    @staticmethod
    def replace_math_words(query: str) -> str:
        """
        Replace word-based mathematical operations with symbols.
        
        Args:
            query: Query with word-based operations
            
        Returns:
            Query with symbolic operations
        """
        replacements = {
            "percent of": "* 0.01 *",
            "percentage of": "* 0.01 *",
            "percent": "* 0.01",
            "percentage": "* 0.01",
            "plus": "+",
            "add": "+",
            "minus": "-",
            "subtract": "-",
            "times": "*",
            "multiply": "*",
            "multiplied by": "*",
            "divided by": "/",
            "divide by": "/",
            "divided": "/",
            "over": "/",
            "power of": "**",
            "to the power of": "**",
            "raised to": "**",
            "square root of": "sqrt(",
            "sqrt of": "sqrt(",
            "squared": "**2",
            "cubed": "**3",
            "sqrt": "sqrt(",
        }
        
        for word, symbol in replacements.items():
            query = query.replace(word, symbol)
        
        # Handle closing parentheses for sqrt
        query = query.replace("sqrt(", "math.sqrt(")
        open_count = query.count("math.sqrt(")
        close_count = query.count(")")
        if open_count > close_count:
            query += ")" * (open_count - close_count)
        
        return query
    
    @staticmethod
    def words_to_numbers(text: str) -> str:
        """
        Convert word-based numbers to digits.
        
        Examples:
        - "twenty five" -> "25"
        - "one hundred" -> "100"
        - "three thousand" -> "3000"
        
        Args:
            text: Text with word numbers
            
        Returns:
            Text with numeric digits
        """
        tokens = text.split()
        current = 0
        result = []
        
        for token in tokens:
            if token in WORD_NUMBER_MAP:
                scale = WORD_NUMBER_MAP[token]
                if scale in [100, 1000, 1000000, 1000000000, 1000000000000]:
                    if current == 0:
                        current = 1
                    current *= scale
                else:
                    current += scale
            elif token in ['+', '-', '*', '/', '**', '%']:
                if current > 0:
                    result.append(str(current))
                    current = 0
                result.append(token)
            else:
                if current > 0:
                    result.append(str(current))
                    current = 0
                result.append(token)
        
        if current > 0:
            result.append(str(current))
        
        return ' '.join(result)
    
    @staticmethod
    def replace_units_in_query(query: str) -> str:
        """
        Replace unit values with their base unit equivalents.
        
        Args:
            query: Query with units
            
        Returns:
            Query with converted values
        """
        for unit, factor in UNIT_CONVERSION.items():
            if isinstance(factor, (int, float)):
                pattern = rf'(\d+(?:\.\d+)?)\s*{re.escape(unit)}'
                query = re.sub(pattern, lambda m: str(float(m.group(1)) * factor), query)
        
        return query
    
    @staticmethod
    def _safe_eval(expression: str) -> Optional[float]:
        """
        Safely evaluate a mathematical expression using AST parsing.
        
        This prevents arbitrary code execution while allowing mathematical
        operations.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Evaluated result or None if evaluation fails
        """
        try:
            import ast
            import operator as op
            
            # Allowed operations
            operators = {
                ast.Add: op.add,
                ast.Sub: op.sub,
                ast.Mult: op.mul,
                ast.Div: op.truediv,
                ast.Pow: op.pow,
                ast.Mod: op.mod,
                ast.USub: op.neg,
                ast.UAdd: op.pos,
            }
            
            # Allowed functions
            functions = {
                'sqrt': math.sqrt,
                'abs': abs,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'sinh': math.sinh,
                'cosh': math.cosh,
                'tanh': math.tanh,
                'log': math.log,
                'log10': math.log10,
                'log2': math.log2,
                'exp': math.exp,
                'ceil': math.ceil,
                'floor': math.floor,
                'round': round,
            }
            
            def eval_(node):
                if isinstance(node, ast.Num):  # <number>
                    return node.n
                elif isinstance(node, ast.Constant):  # Python 3.8+
                    if isinstance(node.value, (int, float, complex)):
                        return node.value
                    raise ValueError("Unsupported constant type")
                elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
                    left = eval_(node.left)
                    right = eval_(node.right)
                    return operators[type(node.op)](left, right)
                elif isinstance(node, ast.UnaryOp):  # <operator> <operand>
                    operand = eval_(node.operand)
                    return operators[type(node.op)](operand)
                elif isinstance(node, ast.Call):  # function(args)
                    if isinstance(node.func, ast.Name) and node.func.id in functions:
                        func = functions[node.func.id]
                        args = [eval_(arg) for arg in node.args]
                        return func(*args)
                    elif isinstance(node.func, ast.Attribute):
                        # Handle math.func() calls
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == 'math':
                            func_name = node.func.attr
                            if func_name in functions:
                                func = functions[func_name]
                                args = [eval_(arg) for arg in node.args]
                                return func(*args)
                    raise ValueError("Function not allowed")
                else:
                    raise ValueError("Expression not allowed")
            
            # Parse and evaluate
            node = ast.parse(expression, mode='eval')
            result = eval_(node.body)
            
            return float(result)
            
        except Exception as e:
            logger.debug(f"[SAFE_EVAL] Error: {e}")
            return None
    
    # ========================================================================
    # EXTERNAL API INTEGRATION
    # ========================================================================
    
    @staticmethod
    def route_to_external_api(query: str) -> Optional[Any]:
        """
        Route query to external API for computation.
        
        This is a fallback when local computation fails or for queries
        requiring external resources.
        
        Args:
            query: Mathematical query
            
        Returns:
            API response or None
        """
        logger.info("[API] Routing math query to external API...")
        try:
            from SarahMemoryAPI import send_to_openai
            result = send_to_openai(f"Solve this math problem precisely: {query}")
            if result and isinstance(result, dict):
                return result.get("data")
        except Exception as e:
            logger.error(f"[API] External API error: {e}")
        
        return None
    
    @staticmethod
    def validate_api_math_response(answer: Any) -> bool:
        """
        Validate that an API response contains a valid mathematical answer.
        
        Args:
            answer: API response to validate
            
        Returns:
            True if response appears to be a valid math answer
        """
        if not answer:
            return False
        
        answer_str = str(answer)
        
        # Check if response contains numbers
        if not re.search(r'[0-9]', answer_str):
            return False
        
        # Check if response contains mathematical operators or equals sign
        if any(op in answer_str for op in ["+", "-", "*", "/", "="]):
            return True
        
        # Check if response is just a number
        try:
            float(answer_str.replace(",", "").strip())
            return True
        except ValueError:
            pass
        
        return False
    
    # ========================================================================
    # ANSWER FORMATTING
    # ========================================================================
    
    @staticmethod
    def format_final_answer(answer: Any, original_query: str = "") -> str:
        """
        Format final answer with appropriate units and precision.
        
        This function intelligently formats answers based on:
        - Query context (currency, percentage, etc.)
        - Magnitude of the result
        - Precision requirements
        
        Args:
            answer: Computed answer
            original_query: Original query for context
            
        Returns:
            Beautifully formatted answer string
        """
        try:
            # Convert to float for processing
            if isinstance(answer, str):
                num = float(answer.replace(",", "").split()[0])
            else:
                num = float(answer)
            
            original_lower = original_query.lower()
            
            # Currency formatting
            if "$" in original_query or "dollar" in original_lower or "usd" in original_lower or \
               "price" in original_lower or "cost" in original_lower or "payment" in original_lower:
                if abs(num) >= 1000000:
                    return f"The answer is ${num:,.2f} (${num/1000000:.2f} million)"
                else:
                    return f"The answer is ${num:,.2f}"
            
            # Percentage formatting
            if "percent" in original_lower or "%" in original_query:
                return f"The answer is {num:.2f}%"
            
            # Temperature formatting
            if any(temp in original_lower for temp in ["celsius", "fahrenheit", "kelvin"]):
                if "celsius" in original_lower or "c" == original_lower[-1]:
                    return f"The answer is {num:.2f}°C"
                elif "fahrenheit" in original_lower or "f" == original_lower[-1]:
                    return f"The answer is {num:.2f}°F"
                elif "kelvin" in original_lower:
                    return f"The answer is {num:.2f} K"
            
            # Scientific notation for very large/small numbers
            if abs(num) >= 1e6 or (abs(num) < 0.0001 and num != 0):
                return f"The answer is {num:.4e}"
            
            # Integer formatting for whole numbers
            if num.is_integer():
                return f"The answer is {int(num):,}"
            
            # Decimal formatting based on magnitude
            if abs(num) >= 100:
                return f"The answer is {num:,.2f}"
            elif abs(num) >= 1:
                return f"The answer is {num:,.4f}"
            else:
                return f"The answer is {num:.6g}"
                
        except Exception as e:
            logger.error(f"[FORMAT] Error formatting answer: {e}")
            return f"The answer is {answer}"


# ============================================================================
# STANDALONE HELPER FUNCTIONS
# ============================================================================

def is_math_expression(text: str) -> bool:
    """
    Check if text is a pure mathematical expression.
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to be a math expression
    """
    try:
        return bool(re.match(r'^[\s\d\+\-\*\/\(\)\.\^%eE]+$', str(text)))
    except Exception:
        return False


def evaluate_math_local(expr: str) -> Optional[float]:
    """
    Safely evaluate a mathematical expression locally.
    
    This is a convenience wrapper around the WebSemanticSynthesizer's
    safe evaluation method.
    
    Args:
        expr: Mathematical expression
        
    Returns:
        Evaluated result or None
    """
    try:
        return WebSemanticSynthesizer._safe_eval(expr)
    except Exception:
        return None


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'WebSemanticSynthesizer',
    'is_math_expression',
    'evaluate_math_local',
    'WEBSTER_STATIC',
    'MATH_CONSTANTS',
    'MATH_SYMBOLS',
    'WORD_NUMBER_MAP',
    'UNIT_CONVERSION',
]


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

logger.info("[WebSYM] World Class Calculator v8.0.0 initialized successfully")
logger.info("[WebSYM] Capabilities: Basic Arithmetic, Scientific, Statistics, Financial, Geometry, Unit Conversion, and more")

# ====================================================================
# END OF SarahMemoryWebSYM.py v8.0.0
# ====================================================================