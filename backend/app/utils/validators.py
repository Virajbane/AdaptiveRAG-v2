import re
from fastapi import HTTPException, status

class InputValidator:
    """Validate and sanitize user inputs"""
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid email format"
            )
        return email.lower()
    
    @staticmethod
    def validate_password(password: str) -> str:
        """Validate password strength"""
        if len(password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters"
            )
        
        if not re.search(r'[A-Z]', password):
            raise HTTPException(status_code=400, detail="Need uppercase letter")
        if not re.search(r'[a-z]', password):
            raise HTTPException(status_code=400, detail="Need lowercase letter")
        if not re.search(r'\d', password):
            raise HTTPException(status_code=400, detail="Need digit")
        if not re.search(r'[!@#$%^&*]', password):
            raise HTTPException(status_code=400, detail="Need special character")
        
        return password
    
    @staticmethod
    def sanitize_string(value: str) -> str:
        """Remove dangerous characters"""
        value = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.IGNORECASE)
        value = re.sub(r'<[^>]+>', '', value)
        value = value.replace('\x00', '')
        return value.strip()