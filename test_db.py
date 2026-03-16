from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql+psycopg2://postgres:postgres@localhost:5432/hro_db"

try:
    engine = create_engine(DATABASE_URL, echo=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("Database connected successfully!")
except Exception as e:
    print("Database connection failed:")
    print(repr(e))