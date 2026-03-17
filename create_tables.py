from database import engine, Base
import models  # noqa: F401

Base.metadata.create_all(bind=engine)

print("Tables created successfully.")