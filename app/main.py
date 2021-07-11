import face_recognition
import base64
import numpy as np
import os
# import pandas as pd
import cv2

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Depends, status
from pydantic import BaseModel, Field
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import List, Optional
from fastapi.openapi.utils import get_openapi

from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

# to get a string like this run:
# openssl rand -hex 32
f = os.popen('openssl rand -hex 32')

SECRET_KEY = f.read()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# easySecret
users_db = {
    "easyClient": {
        "username": "easyClient",
        "full_name": "Camilo Aguilar",
        "email": "camilo.aguilar@easydata.com.co",
        "hashed_password": "$2b$12$k8.IWo8c0qJM.ANzndbpPuNwKNIkK7bKzn6xxSAxc9lCU.bWGVFh.",
        "scope": 'admin',
        "disabled": False,
    }
}


# make a API
class Body(BaseModel):
    encoded_id: str = Field(..., example='encoded_photo_id_base64')
    encoded_photo: str = Field(..., example='encoded_selfie_base64')
    id_number: int = Field(..., example=1013596884)


class Hdr(BaseModel):
    content_type: str


class Biometric(BaseModel):
    # isBase64Encoded: Optional[bool] = True
    headers: Hdr
    body: Body


# easybio return model
class img_return(BaseModel):
    filename: str = Field(..., example="image.jpg")
    dimensions: str = Field(..., example="720p")


class result_return(BaseModel):
    is_same_person: bool
    face_distances: float = Field(..., example=0.2)
    similarity_percentage: float = Field(..., example=98.3)


class easybio_v2(BaseModel):
    user: str = Field(..., example='easyUser')
    ip_client: str = Field(..., example="0.0.0.0")
    id: img_return
    photo: img_return
    result: result_return


# Token return model
class Token(BaseModel):
    access_token: str = Field(..., example="YOUR_EASYBIO_ACCESS_TOKEN")
    token_type: str = Field(..., example='bearer')
    expire_minutes: int = Field(..., example=60)


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str = Field(..., example='my_user')
    email: Optional[str] = Field(None, example='myemail@example.com')
    full_name: Optional[str] = None
    scope: str
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(
    title='easybio API',
    description='Esta es una descripción',
    version='1.0',
    docs_url='/docs',
    redoc_url='/redoc'
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="easybio API",
        version="1.0.3",
        description="Este servicio web le permite, de forma programática, verificar la identidad de sus usuarios.",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://i.ibb.co/CBJMqFd/logo-easydata-cropped.jpg"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(db_users, username: str, password: str):
    user = get_user(db_users, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# ***************
# API v2
# ***************
@app.post("/api/v2/oauth2", response_model=Token, tags=['oauth'])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token,
            "expire_minutes": ACCESS_TOKEN_EXPIRE_MINUTES,
            "token_type": "bearer"}


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token,
            "expire_minutes": ACCESS_TOKEN_EXPIRE_MINUTES,
            "token_type": "bearer"}


@app.post(
    "/api/v2/form/easybio/",
    tags=['easybio'],
    summary="Validación de identidad [form]",
    response_model=easybio_v2
)
async def easybio_upload_file(request: Request, files: List[UploadFile] = File(...),
                              clurrent_user: User = Depends(get_current_active_user)):
    """
    Envía una petición de validación de identidad con las siguientes consideraciones:

    - **Peso**: El peso máximo de cada imagen debe ser de `1Mb`. Se sugiere un peso `<= 100Kb` para incrementar rendimiento.
    - **Dimensiones**: Mínimo `480 * 480` pixeles.
    - **Formato**: Se sugiere formato de compresión _jpeg_
    - **Orientación**: Las imágenes deben enviarse con correcta orientación. Las imágenes giradas no podrán ser procesadas y devolverán un `error 440`
    """
    # Camilo, Eduardo
    ip_permitted = ['186.84.22.231', '181.58.182.194', '127.0.0.1']
    client_host = request.client.host

    if client_host not in ip_permitted:
        raise HTTPException(status_code=401, detail="Unauthorized IP " + client_host)

    resize_rate = 1 / 4
    contents_1 = await files[0].read()
    image_1 = np.asarray(bytearray(contents_1), dtype="uint8")
    image_1 = cv2.imdecode(image_1, cv2.IMREAD_COLOR)
    img_dimensions_1 = str(image_1.shape)
    image_1 = cv2.resize(image_1, (0, 0), fx=resize_rate, fy=resize_rate)
    file_name_1 = files[0].filename + '_' + client_host + '.jpg'
    cv2.imwrite('./img/temp/' + file_name_1, image_1, [cv2.IMWRITE_JPEG_QUALITY, 50])

    contents_2 = await files[1].read()
    image_2 = np.asarray(bytearray(contents_2), dtype="uint8")
    image_2 = cv2.imdecode(image_2, cv2.IMREAD_COLOR)
    img_dimensions_2 = str(image_2.shape)
    image_2 = cv2.resize(image_2, (0, 0), fx=resize_rate, fy=resize_rate)
    file_name_2 = files[1].filename + '_' + client_host + '.jpg'
    cv2.imwrite('./img/temp/' + file_name_2, image_2, [cv2.IMWRITE_JPEG_QUALITY, 50])

    # facial recognition
    try:
        face_locations = face_recognition.face_locations(image_1[:, :, ::-1])
        known_encoding = face_recognition.face_encodings(image_1, face_locations)[0]
    except Exception as e:
        raise HTTPException(status_code=440, detail="Oops! no he detectado ningún rostro en el id")

    try:
        face_locations = face_recognition.face_locations(image_2[:, :, ::-1])
        unknown_encoding = face_recognition.face_encodings(image_2, face_locations)[0]
    except Exception as e:
        raise HTTPException(status_code=440, detail="Oops! no he detectado ningún en la fotografía")

    face_distances = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
    similarity = np.min([1 - face_distances, 0.60]) / (0.6 + np.random.uniform(0, 0.01, 1)[0]) * 100
    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    results = results[0]

    return {
        "user": current_user.username,
        'ip_client': client_host,
        'id': {
            'filename': files[0].filename,
            'dimensions': img_dimensions_1,
        },
        'photo': {
            'filename': files[1].filename,
            'dimensions': img_dimensions_2,
        },
        'result': {
            'is_same_person': str(results),
            'face_distances': face_distances,
            'similarity_percentage': similarity
        }
    }


# ***************
# API v1
# ***************
@app.post(
    "/api/v1/form/easybio/",
    tags=['easybio'],
    summary="Validación de identidad [form]",
)
async def easybio_upload_file(request: Request, files: List[UploadFile] = File(...)):
    """
    Envía una petición de validación de identidad con las siguientes consideraciones:

    - **Peso**: El peso máximo de cada imagen debe ser de `1Mb`. Se sugiere un peso `<= 100Kb` para incrementar rendimiento.
    - **Dimensiones**: Mínimo `480 * 480` pixeles.
    - **Formato**: Se sugiere formato de compresión _jpeg_
    - **Orientación**: Las imágenes deben enviarse con correcta orientación. Las imágenes giradas no podrán ser procesadas y devolverán un `error 440`
    """
    # Camilo, Eduardo
    ip_permitted = ['186.84.22.231', '181.58.182.194', '127.0.0.1']
    client_host = request.client.host

    if client_host not in ip_permitted:
        raise HTTPException(status_code=401, detail="Unauthorized IP " + client_host)

    resize_rate = 1 / 4
    contents_1 = await files[0].read()
    image_1 = np.asarray(bytearray(contents_1), dtype="uint8")
    image_1 = cv2.imdecode(image_1, cv2.IMREAD_COLOR)
    img_dimensions_1 = str(image_1.shape)
    image_1 = cv2.resize(image_1, (0, 0), fx=resize_rate, fy=resize_rate)
    file_name_1 = files[0].filename + '_' + client_host + '.jpg'
    cv2.imwrite('./img/temp/' + file_name_1, image_1, [cv2.IMWRITE_JPEG_QUALITY, 50])

    contents_2 = await files[1].read()
    image_2 = np.asarray(bytearray(contents_2), dtype="uint8")
    image_2 = cv2.imdecode(image_2, cv2.IMREAD_COLOR)
    img_dimensions_2 = str(image_2.shape)
    image_2 = cv2.resize(image_2, (0, 0), fx=resize_rate, fy=resize_rate)
    file_name_2 = files[1].filename + '_' + client_host + '.jpg'
    cv2.imwrite('./img/temp/' + file_name_2, image_2, [cv2.IMWRITE_JPEG_QUALITY, 50])

    # facial recognition
    try:
        face_locations = face_recognition.face_locations(image_1[:, :, ::-1])
        known_encoding = face_recognition.face_encodings(image_1, face_locations)[0]
    except Exception as e:
        raise HTTPException(status_code=440, detail="Oops! no he detectado ningún rostro en el id")

    try:
        face_locations = face_recognition.face_locations(image_2[:, :, ::-1])
        unknown_encoding = face_recognition.face_encodings(image_2, face_locations)[0]
    except Exception as e:
        raise HTTPException(status_code=440, detail="Oops! no he detectado ningún en la fotografía")

    face_distances = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
    similarity = np.min([1 - face_distances, 0.60]) / (0.6 + np.random.uniform(0, 0.01, 1)[0]) * 100
    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    results = results[0]

    return {
        'ip client': client_host,
        'id': {
            'filename': files[0].filename,
            'dimensions': img_dimensions_1,
        },
        'photo': {
            'filename': files[1].filename,
            'dimensions': img_dimensions_2,
        },
        'result': {
            'is_same_person': str(results),
            # 'face_distances': face_distances,
            'similarity_percentage': similarity
        }
    }


@app.post(
    "/api/v1/b64/easybio/",
    tags=['easybio'],
    summary="Validación de identidad [base64]",
    response_description="Resultado de la validación"
    # description = "Envía foto de documento de identidad y foto tomada directamente a la persona para validar id"
)
def easybio_lite(request: Request, biofile: Biometric):
    """
    Envía una petición de validación de identidad con la siguiente información:

    - **isBase64Encoded**: [Optional] Valor booleano que informa si las imagenes a validar están codificadas en _Base64_. easybio manejará la conversión de nuevo a binario
    - **headers**:
        - **content-type**: Establecer encabezado de tipo de contenido, por ejemplo como image/jpeg
    - **body**:
        - **encoded_id**: foto del anverso del documento de identidad, codificada bajo el estándar _base64_
        - **encoded_photo**: foto del individuo sobre el cual se realizará validación de identidad, codificada bajo el estándar _base64_
        - **id_number**: número de identificación asociada al documento de identidad.
    """
    # Camilo, Eduardo
    ip_permitted = ['127.0.0.1', '186.84.22.231', '181.58.182.194']
    client_host = request.client.host

    if client_host not in ip_permitted:
        raise HTTPException(status_code=401, detail="Unauthorized IP " + client_host)

    # decoding
    path_know_decoded = './img/temp/known_decoded'
    path_unknow_decoded = './img/temp/unknown_decoded'

    try:
        with open(path_know_decoded, 'wb') as image_save:
            image_save.write(base64.b64decode(bytes(biofile.body.encoded_id)))

        with open(path_unknow_decoded, 'wb') as image_save:
            image_save.write(base64.b64decode(bytes(biofile.body.encoded_photo)))

        # decoded with numpy
        # nparr = np.fromstring(biofile.body.encoded_id, np.uint8)
        # img = cv2.imdecode(biofile.body.encoded_id, cv2.IMREAD_COLOR)

        # run face recognition
        known_image = face_recognition.load_image_file(path_know_decoded)
        unknown_image = face_recognition.load_image_file(path_unknow_decoded)

        face_locations = face_recognition.face_locations(known_image)
        known_encoding = face_recognition.face_encodings(known_image, face_locations)[0]

        face_locations = face_recognition.face_locations(unknown_image)
        unknown_encoding = face_recognition.face_encodings(unknown_image, face_locations)[0]

        face_distances = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
        similarity = np.min([1 - face_distances, 0.60]) / (0.6 + np.random.uniform(0, 0.01, 1)[0]) * 100
        results = face_recognition.compare_faces([known_encoding], unknown_encoding)
        results = results[0]

        return {'status': 'ok',
                'face_distance': face_distances,
                'match_%': similarity,
                'is_same_person': str(results)}

    except Exception as e:
        return {'status': 'error',
                'validation': False,
                'msj': 'error en decodificación base64',
                'exception': e}
