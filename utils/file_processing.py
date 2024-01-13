from uuid import uuid4
import os
import ffmpeg

# Debido a que el servidor recibe un "Archivo Temporal" cuando se sube un archivo, se debe de crear un archivo en disco
# Primero se lee el archivo temporal, se establece la posicion en el item 0
# Se genera un nuevo archivo con un nombre "X", este archivo aún es del tipo original (mp3, m4a, ogg, etc)
# Una vez guardado en disco el archivo, se convierte en WAV, utilizando ffmpeg
# Al final se devuelve la ruta de dicho archivo creado
def create_wav_file(file, tmp_path = "./tmp"):
    audio_file = file.file
    audio_file.seek(0)
    file_extension = file.filename.split(".")[-1]
    file_name = uuid4().hex[:12] + "." + file_extension
    created_audio_file = os.path.join(tmp_path,file_name)
    with open(created_audio_file, "wb") as af:
        af.write(audio_file.read())
    
    wav_file_name = created_audio_file.replace(file_extension, "wav")
    
    ffmpeg.input(created_audio_file).output(wav_file_name).run()
    
    return wav_file_name