from fastapi import APIRouter
from api.routes import upload, pipeline, chat, artifacts

router = APIRouter()
router.include_router(upload.router,    prefix="/dataset",   tags=["Dataset"])
router.include_router(pipeline.router,  prefix="/pipeline",  tags=["Pipeline"])
router.include_router(chat.router,      prefix="/analyst",   tags=["Chat Analyst"])
router.include_router(artifacts.router, prefix="/artifacts", tags=["Artifacts"])
