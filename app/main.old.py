import face_recognition
import base64
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, status
from pydantic import BaseModel
from typing import List, Optional
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm


# make a API
class Body(BaseModel):
    encoded_id: str
    encoded_photo: str
    id_number: int


class Hdr(BaseModel):
    content_type: str


class Biometric(BaseModel):
    # isBase64Encoded: Optional[bool] = True
    headers: Hdr
    body: Body


app = FastAPI()


@app.post(
    "/api/form/easybio/",
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
    ip_permitted = ['186.84.22.231', '181.58.182.194']
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
    "/api/b64/easybio/",
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



fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    },
    "alice": {
        "username": "alice",
        "full_name": "Alice Wonderson",
        "email": "alice@example.com",
        "hashed_password": "fakehashedsecret2",
        "disabled": True,
    },
}

def fake_hash_password(password: str):
    return "fakehashed" + password


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def fake_decode_token(token):
    # This doesn't provide any security at all
    # Check the next version
    user = get_user(fake_users_db, token)
    return user


async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_decode_token(token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = fake_users_db.get(form_data.username)
    if not user_dict:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    user = UserInDB(**user_dict)
    hashed_password = fake_hash_password(form_data.password)
    if not hashed_password == user.hashed_password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    return {"access_token": user.username, "token_type": "bearer"}


@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user
