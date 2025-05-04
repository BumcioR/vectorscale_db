# 1. Importy
import numpy as np

# Typowanie
from typing import List, Optional

# Zewnętrzne biblioteki
from sqlalchemy.engine import URL
from sqlalchemy import create_engine, Integer, String, Float, Boolean, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm

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
    __abstract__ = True

# 4. Definicje modeli
class Images(Base):
    __tablename__ = "images"
    VECTOR_LENGTH = 512

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_path: Mapped[str] = mapped_column(String(256))
    image_embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_LENGTH))


class Games(Base):
    __tablename__ = "games"
    VECTOR_LENGTH = 512

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(String(4096))
    windows: Mapped[bool] = mapped_column(Boolean)
    linux: Mapped[bool] = mapped_column(Boolean)
    mac: Mapped[bool] = mapped_column(Boolean)
    price: Mapped[float] = mapped_column(Float)
    game_description_embedding: Mapped[List[float]] = mapped_column(Vector(VECTOR_LENGTH))


# 5. Inicjalizacja silnika i modelu
engine = create_engine(db_url)
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

def generate_embeddings(text: str) -> List[float]:
    return model.encode(text).tolist()

# 6. Wstawianie danych
def insert_image(engine, image_path: str, image_embedding: List[float]):
    with Session(engine) as session:
        image = Images(image_path=image_path, image_embedding=image_embedding)
        session.add(image)
        session.commit()

def insert_games(engine, dataset):
    with Session(engine) as session:
        for game in tqdm(dataset, total=len(dataset)):
            name = game["Name"]
            description = game["About the game"] or ""
            windows, linux, mac = game["Windows"], game["Linux"], game["Mac"]
            price = game["Price"]

            if name and description and price is not None:
                embedding = generate_embeddings(description)
                game_obj = Games(
                    name=name,
                    description=description[:4096],
                    windows=windows,
                    linux=linux,
                    mac=mac,
                    price=price,
                    game_description_embedding=embedding
                )
                session.add(game_obj)
        session.commit()

# 7. Zapytania
def find_game(engine, game_description: str, windows: Optional[bool] = None,
              linux: Optional[bool] = None, mac: Optional[bool] = None,
              price: Optional[float] = None):
    game_embedding = generate_embeddings(game_description)
    with Session(engine) as session:
        query = select(Games).order_by(Games.game_description_embedding.cosine_distance(game_embedding))

        if price is not None:
            query = query.filter(Games.price <= price)
        if windows is not None:
            query = query.filter(Games.windows == windows)
        if linux is not None:
            query = query.filter(Games.linux == linux)
        if mac is not None:
            query = query.filter(Games.mac == mac)

        result = session.execute(query, execution_options={"prebuffer_rows": True})
        return result.scalars().first()

def find_k_images(engine, k: int, original_image: Images):
    with Session(engine) as session:
        result = session.execute(
            select(Images)
            .order_by(Images.image_embedding.cosine_distance(original_image.image_embedding))
            .limit(k),
            execution_options={"prebuffer_rows": True}
        )
        return [row[0] for row in result]
    
def find_images_with_similarity_score_greater_than(engine, similarity_score: float, original_image: Images) -> list[Images]:
    with Session(engine) as session:
        result = session.execute(
            select(Images)
            .filter(Images.image_embedding.cosine_similarity(original_image.image_embedding) > similarity_score),
            execution_options={"prebuffer_rows": True}
        )
        return result.scalars().all()

# 8. Główna logika
if __name__ == "__main__":
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    print("Tabele zostały utworzone.")

    # Wstawianie obrazów testowych
    N = 100
    for i in range(N):
        path = f"image_{i}.jpg"
        vector = np.random.rand(512).tolist()
        insert_image(engine, path, vector)
        if i % 10 == 0:
            print(f"Wstawiono {i} obrazów...")

    print("Obrazy testowe zostały wstawione.")

    # Pobieranie pierwszego obrazu
    with Session(engine) as session:
        first_image = session.query(Images).first()

    if first_image:
        similar_images = find_k_images(engine, 10, first_image)
        print("\nNajbardziej podobne obrazy:")
        for sim in similar_images:
            print(sim.image_path)
    else:
        print("Nie znaleziono żadnych obrazów w bazie.")

    # Pobieranie danych gier
    columns_to_keep = ["Name", "Windows", "Linux", "Mac", "About the game", "Supported languages", "Price"]
    dataset = load_dataset("FronkonGames/steam-games-dataset")
    dataset = dataset["train"].select_columns(columns_to_keep).select(range(1000))  # startowo 1000 rekordów

    insert_games(engine, dataset)
    print("Gry zostały wstawione.")

    print("\nTestujemy wyszukiwanie gry:")
    game = find_game(engine, game_description="zombie survival game", price=20, windows=True)
    if game:
        print(f"Nazwa: {game.name}")
        print(f"Opis: {game.description}")
    else:
        print("Nie znaleziono gry.")
