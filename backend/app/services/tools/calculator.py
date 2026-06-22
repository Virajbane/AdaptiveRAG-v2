import re
import math
from typing import Dict

class CalculatorTool:
    """Safe math calculator"""
    
    ALLOWED_FUNCTIONS = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'exp': math.exp,
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
    }
    
    ALLOWED_CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
    }
    
    async def calculate(self, expression: str) -> Dict:
        """
        Calculate math expression safely
        
        Supports: +, -, *, /, **, (), functions, constants
        """
        try:
            # Validate expression
            if not self._is_safe(expression):
                return {
                    "result": None,
                    "error": "Invalid characters in expression",
                    "expression": expression
                }
            
            # Build safe namespace
            namespace = {
                '__builtins__': {},
                **self.ALLOWED_FUNCTIONS,
                **self.ALLOWED_CONSTANTS
            }
            
            # Evaluate
            result = eval(expression, namespace)
            
            return {
                "result": result,
                "expression": expression,
                "error": None
            }
        except ZeroDivisionError:
            return {
                "result": None,
                "error": "Division by zero",
                "expression": expression
            }
        except Exception as e:
            return {
                "result": None,
                "error": str(e),
                "expression": expression
            }
    
    def _is_safe(self, expression: str) -> bool:
        """Check if expression is safe to evaluate"""
        # Only allow numbers, operators, functions, constants, parens
        allowed = r'^[0-9+\-*/().,\s\w]*$'
        return bool(re.match(allowed, expression))

# Global instance
calculator_tool = CalculatorTool()