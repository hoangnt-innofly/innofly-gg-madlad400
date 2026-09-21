"""Run the MT API:  py -3.12 -m uvicorn app.main:app --host 0.0.0.0 --port 8002"""

from app.core.config import get_settings
from app.main import app

settings = get_settings()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )
