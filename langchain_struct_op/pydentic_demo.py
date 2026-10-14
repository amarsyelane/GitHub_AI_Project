from datetime import datetime
from pydantic import BaseModel, Field, PositiveInt, EmailStr
from typing import Optional

class User(BaseModel):
    name: str="Unknown"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0.0, lt=10.0, default=5.0,description="Cumulative Grade Point Average")  # CGPA must be between 0.0 and 4.0

new_user = {'name': 'John Doe', 'age': 30, 'email': 'john.doe@example.com', 'cgpa': 9}

student = User(**new_user)
print(student)

'''from pydantic import BaseModel, EmailStr

class User(BaseModel):
    email: EmailStr

# Success
user = User(email="contact@example.com")
print(user.email) #> contact@example.com

# Fails with ValidationError
try:
    User(email="invalid-email")
except Exception as e:
    print(e)'''