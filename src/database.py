import os
import threading
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

# Global lock for database operations
db_lock = threading.Lock()

# Global engine instance (create once)
_engine = None
_SessionLocal = None


def init_db():
    """Initialize database (thread-safe)"""
    global _engine, _SessionLocal

    with db_lock:
        if _engine is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            data_folder = os.path.join(base_dir, "data")

            os.makedirs(data_folder, exist_ok=True)

            db_path = os.path.join(data_folder, "detected_objects.db")

            _engine = create_engine(
                f"sqlite:///{db_path}",
                connect_args={"check_same_thread": False}  # Allow multi-threading
            )
            Base.metadata.create_all(_engine, checkfirst=True)
            _SessionLocal = sessionmaker(bind=_engine)

            print(f"Database initialized at: {db_path}")

    return _engine


class DetectedObject(Base):
    __tablename__ = "detected_objects"

    id = Column(Integer, primary_key=True)
    box = Column(Integer)
    Class = Column(String)
    color = Column(String)
    scores = Column(Float)
    box_coordinates = Column(String)


def save_objects_data_to_database(objects_data):
    if not objects_data:
        return

    # Initialize DB if needed
    init_db()

    # Use lock to prevent concurrent database writes
    with db_lock:
        Session = sessionmaker(bind=_engine)
        session = Session()

        try:
            for obj_data in objects_data:
                detected_obj = DetectedObject(
                    box=obj_data["Box"],
                    Class=obj_data["Class"],
                    color=obj_data["Color"],
                    scores=obj_data["Scores"],
                    box_coordinates=obj_data["Box Coordinates"],
                )
                session.add(detected_obj)

            session.commit()
            print(f"Saved {len(objects_data)} objects to database")
        except Exception as e:
            session.rollback()
            print(f"Error saving to database: {e}")
        finally:
            session.close()