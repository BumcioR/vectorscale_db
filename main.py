
# 1. Import bibliotek
from sqlalchemy.engine import URL
from sqlalchemy import create_engine, Integer, String, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from pgvector.sqlalchemy import Vector
from typing import List
import numpy as np

# 2. Konfiguracja URL do bazy danych
db_url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password="password",
    host="localhost",
    port=5555,
    database="similarity_search_service_db"
)

# 3. Klasa bazowa
class Base(DeclarativeBase):
    __abstract__ = True  # nie tworzona jako osobna tabela

# 4. Definicja tabeli "images"
class Images(Base):
    __tablename__ = "images"
    VECTOR_LENGTH = 512

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_path: Mapped[str] = mapped_column(String(256))
    image_embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_LENGTH))

# 5. Tworzenie silnika do połączenia z bazą danych
engine = create_engine(db_url)

# 6. Tworzenie tabeli w bazie danych
if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("Tabela 'images' zostala utworzona.")

    # Wstawianie danych
    def insert_image(engine, image_path: str, image_embedding: list[float]):
        with Session(engine) as session:
            image = Images(image_path=image_path, image_embedding=image_embedding)
            session.add(image)
            session.commit()

    N = 100
    for i in range(N):
        path = f"image_{i}.jpg"
        vector = np.random.rand(512).tolist()
        insert_image(engine, path, vector)
        if i % 10 == 0:
            print(f"Wstawiono {i} obrazów...")

    print("Dane testowe zostały wstawione.")

    # Pobieranie pierwszego obrazu
    with Session(engine) as session:
        first_image = session.query(Images).first()

    # Wyszukiwanie najbardziej podobnych
    def find_k_images(engine, k: int, original_image: Images):
        with Session(engine) as session:
            result = session.execute(
                select(Images)
                .order_by(Images.image_embedding.cosine_distance(original_image.image_embedding))
                .limit(k),
                execution_options={"prebuffer_rows": True}
            )
            return [row[0] for row in result]

    similar_images = find_k_images(engine, 10, first_image)

    for sim in similar_images:
        print(sim.image_path)
