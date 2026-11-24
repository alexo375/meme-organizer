from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from meme_organizer.curation.store import CurationStore
from meme_organizer.schemas.curation import CuratedRecord


class CurationAppFactory:
    """Фабрика для создания FastAPI приложения доразметки."""

    @classmethod
    def create_app(cls, silver_dir: Path, gold_dir: Path) -> FastAPI:
        """Создает и настраивает экземпляр FastAPI приложения."""
        app = FastAPI(title="Meme Organizer Curation API")
        store = CurationStore(silver_dir, gold_dir)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        static_dir = Path(__file__).parent / "static"

        @app.get("/", response_class=HTMLResponse)
        def index() -> str:
            return (static_dir / "index.html").read_text(encoding="utf-8")

        @app.get("/api/memes")
        def get_memes() -> list[CuratedRecord]:
            return store.get_all()

        @app.post("/api/memes/{file_hash}")
        def update_meme(file_hash: str, payload: CuratedRecord) -> dict[str, str]:
            if file_hash != payload.file_hash:
                raise HTTPException(status_code=400, detail="Mismatched file hash")
            store.update_record(payload)
            return {"status": "ok"}

        @app.delete("/api/memes/{file_hash}")
        def delete_meme(file_hash: str) -> dict[str, str]:
            deleted = store.delete_record(file_hash, delete_files=True)
            if not deleted:
                raise HTTPException(status_code=404, detail="Meme not found")
            return {"status": "deleted"}

        @app.post("/api/export-gold")
        def export_gold() -> dict[str, int]:
            count = store.export_gold()
            return {"exported_count": count}

        app.mount("/media", StaticFiles(directory=str(silver_dir)), name="media")
        return app
