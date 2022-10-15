from pydantic import BaseModel, Field
from typing import  Optional


class MessageSchema(BaseModel):
    pass

    class Config:
        schema_extra = {

            }


def ResponseModel(data, message):
    return {
        'data': [data],
        'code': 200,
        'message': message
    }


def ErrorResponseModel(error, code, message):
    return {'error': error, 'code': code, 'message': message}