from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    username: str
    name: str | None = None
    role: str | None = None
    department: str | None = None