from fastapi import APIRouter,WebSocket,WebSocketDisconnect,Body,Depends
from .ws import ConnectionManager

from fastapi import  File, UploadFile



router_rest = APIRouter()

@router_rest.post('/files')
async def create_file(file:bytes = File()):
    return {'file_size':len(file)}

@router_rest.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}



