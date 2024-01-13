from fastapi import FastAPI, UploadFile
from starlette.responses import JSONResponse
from utils.file_processing import create_wav_file
import numpy as np
import scipy
import uvicorn
import os

FILES_PATH = "./tmp"
# Valor minimo para obtener las frecuencias caracteristicas
AMP_THRESH = 0.5
# Numero de items a comparar (audio vs clases)
CHAR_NUMB = 100

# Valores numericos de las clases, calculados previamente
DATA_CLASS_A =  [ float(x) for x in open("./data/class_a.txt", "r", encoding="utf-8").read().split(",") ] 
DATA_CLASS_E =  [ float(x) for x in open("./data/class_e.txt", "r", encoding="utf-8").read().split(",") ] 
DATA_CLASS_I =  [ float(x) for x in open("./data/class_i.txt", "r", encoding="utf-8").read().split(",") ] 
DATA_CLASS_O =  [ float(x) for x in open("./data/class_o.txt", "r", encoding="utf-8").read().split(",") ] 
DATA_CLASS_U =  [ float(x) for x in open("./data/class_u.txt", "r", encoding="utf-8").read().split(",") ] 

DATA_CLASSES = [ 
    { 
        "class": "a", "data": DATA_CLASS_A 
    },
    { 
        "class": "e", "data": DATA_CLASS_E
    },
    { 
        "class": "i", "data": DATA_CLASS_I
    },
    { 
        "class": "o", "data": DATA_CLASS_O
    },
    { 
        "class": "u", "data": DATA_CLASS_U
    }
]

if not os.path.isdir(FILES_PATH):
    os.mkdir(FILES_PATH)

app = FastAPI()

# Ruta de prueba
@app.get("/api")
def test_api():
    return JSONResponse({ "message": "Ok"}, status_code=200)

# Ruta unica donde se procesa el archivo de audio
@app.post("/process_file")
async def process_audio_file(file: UploadFile):
    
    # Se crea un archivo WAV para poder visualizar los componentes de audio
    wav_file = create_wav_file(file, FILES_PATH)
    
    # Se lee el archivo y se obtiene el sampling rate y un arreglo de dimension N con los componentes de audio
    sr , data = scipy.io.wavfile.read(wav_file)
    
    # Si el arreglo no es de 1-D, se eliminan todos los canales (otras dimensiones) extras
    data = data.reshape(-1)
    
    # Filtramos unicamente las componentes que tengan valor (no sean silencio)
    data = [ x for x in data if x != 0 ]
    
    # Calcula la FFT de una señal compuesta puramente por valores reales
    yf = scipy.fft.rfft(data)
    # Devuelve las frecuencias de muestreo de la DFT
    xf = scipy.fft.rfftfreq(len(data), 1 / sr)
    
    # Obtenemos unicamente 1024 items (por comodidad, ya que con 2048, el proceso se hace muy lento)
    yf = np.resize(yf, (1024, ))
    xf = np.resize(xf, (1024, ))
    
    # Obtenemos el espectro de amplitud, calculando el valor absoluto de la FFT
    amp_spec = np.abs(yf)
    # Normalizamos o linealizamos, dividiendo los valores del espectro entre el mas grande de este
    # --> X / max(amp_spec)
    amp_spec = amp_spec / np.max(amp_spec)
    
    # Obtenemos las frecuencias de muestreo en las posiciones donde amp_spec >= AMP_THRESH
    char_freqs = xf[ amp_spec >= AMP_THRESH ]
    
    # Obtenemos los N items
    char_freqs = np.resize(char_freqs, (CHAR_NUMB,))
    
    # Realizamos la similitud PEARSON, debido a que son +2 clases, comparando los valores de char_freq con los valores guardados en los archivos
    # Se obtiene un arreglo de N items con un valor de 0 a 1, indicando la similitud, entre las 5 clases
    pearson_simil = [ np.corrcoef(char_freqs, c["data"] )[0][1] for c in DATA_CLASSES]
    
    # Se obtiene el item con mayor valor
    max_value = np.max(pearson_simil)
    # Se obtiene el item de dicho valor
    idx_max_value = pearson_simil.index(max_value)
    # Seleccionamos la clase correspondiente a dicho indice
    selected_class = DATA_CLASSES[idx_max_value]
    
    # Hacemos un resize a 2048 items del archivo original (solo para graficar)
    ydata = np.resize(data, (2048,))
    
    # Obtenemos los valores de la grafica de audio
    # X: desde 0 hasta la duracion del audio
    # Y: Valores de audio
    audio_data = [ { "x": float(x), "y": float(y) } for x, y in zip(range(len(ydata)), ydata) ]
    
    # Obtenemos los valores de frontera
    min_ydata = float(np.min(ydata)) 
    max_ydata = float(np.max(ydata))
    max_xdata = len(ydata)
    
    # El JSON de respuesta
    fft_response = {
        "message": "Se han obtenido los calculos correctamente",
        "class": selected_class["class"].upper(),
        "audio_data": audio_data,
        "min_ydata": min_ydata,
        "max_ydata": max_ydata,
        "max_xdata": max_xdata,
        "audio_freqs": char_freqs.tolist(),
        "class_freqs": selected_class["data"]
    }
    
    files_to_delete = os.listdir(FILES_PATH)
    
    # Limpiamos los archivos, ya que ocupan espacio y se generan muchos
    for f in files_to_delete:
        os.remove(os.path.join(FILES_PATH, f))
    
    return JSONResponse(fft_response, status_code=200)


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")