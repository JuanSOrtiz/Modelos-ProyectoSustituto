# Proyecto sustituto Modelos 1

## Miembro del grupo
Juan Sebastián Ortiz Tangarife, CC 1001366811, Ingeniería de Sistemas


## Enlace al Dataset utilizado
https://www.kaggle.com/competitions/playground-series-s4e2/overview

### Entrega 1
LOS DATOS ESTÁN DISPONIBLES EJECUTANDO LAS CELDAS QUE ESTÁN DENTRO DE "CARGAR LOS ARCHIVOS DESDE KAGGLE". Te pide un user y un key que están disponibles en kaggle -> settings -> Api -> Create new token. Eso descarga un Json con el user y la key


### Entrega 2
Los CSV ya se encuentran dentro del repositorio, sin embargo también se pueden obtener a través del archivo 01 Generate Data dentro de la carpeta Entrega 2.
Para correr los scripts, solo se debe ejecutar el archivo 02_run_scripts que se encuentra dentro de la carpeta Entrega 2.
El archivo Dockerfile también se encuentra adjunto dentro de la carpeta Entrega 2


### Entrega 2
Todos los CSV ya se encuentran dentro del repositorio en la carpeta Entrega 3.
Para correr los scripts primero se debe hacer build al container
`docker build -t modelos-entrega3 .`

Luego se deberá correr el container
`docker run -p 5000:5000 modelos-entrega3`

Finalmente, correr el archivo client.py que se encuentra dentro de la carpeta Entrega 3.

