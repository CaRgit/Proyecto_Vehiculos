import cv2
import numpy as np
import os
import time

def detectar_carriles(frame):
    # Convertir el frame a escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque gaussiano para reducir el ruido
    desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Detectar bordes usando el operador de Canny
    bordes = cv2.Canny(desenfoque, 50, 150)
    
    # Definir una región de interés (ROI) trapezoidal
    altura, ancho = frame.shape[:2]
    puntos_roi = np.array([[(ancho-ancho*0.95, altura*0.95), (ancho * 0.35, altura * 0.75), (ancho * 0.65, altura * 0.75), (ancho*0.95, altura*0.95)]], dtype=np.int32)

    # Dibujar la zona de interés con una línea discontínua verde
    cv2.polylines(frame, [puntos_roi], isClosed=True, color=(0, 255, 0), thickness=1)
    mascara = np.zeros_like(gris)
    cv2.fillPoly(mascara, puntos_roi, 255)
    imagen_roi = cv2.bitwise_and(bordes, mascara)

    # Detectar líneas usando la transformada de Hough
    lineas = cv2.HoughLinesP(imagen_roi, 1, np.pi/180, 50, minLineLength=50, maxLineGap=100)
    
    # Separar las líneas en grupos izquierda y derecha
    lineas_izquierda = []
    lineas_derecha = []
    if lineas is not None:  # Verificar si se detectaron líneas
        for linea in lineas:
            x1, y1, x2, y2 = linea[0]
            pendiente = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita
            if pendiente < 0:
                lineas_izquierda.append(linea[0])
            else:
                lineas_derecha.append(linea[0])

    # Promediar las coordenadas de las líneas izquierda y derecha si se detectaron
    if lineas_izquierda:
        promedio_izquierda = np.mean(lineas_izquierda, axis=0, dtype=np.int32)
        x1, y1, x2, y2 = promedio_izquierda
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 5)

        # Extender las líneas
        pendiente = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita
        if pendiente != 0:  # Evitar división por cero
            interseccion = y1 - pendiente * x1
            y_superior = 0
            y_inferior = altura
            x_superior = int((y_superior - interseccion) / pendiente)
            x_inferior = int((y_inferior - interseccion) / pendiente)
            cv2.line(frame, (x_superior, y_superior), (x_inferior, y_inferior), (255, 0, 0), 1)

    if lineas_derecha:
        promedio_derecha = np.mean(lineas_derecha, axis=0, dtype=np.int32)
        x1, y1, x2, y2 = promedio_derecha
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 5)

        # Extender las líneas
        pendiente = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita
        if pendiente != 0:  # Evitar división por cero
            interseccion = y1 - pendiente * x1
            y_superior = 0
            y_inferior = altura
            x_superior = int((y_superior - interseccion) / pendiente)
            x_inferior = int((y_inferior - interseccion) / pendiente)
            cv2.line(frame, (x_superior, y_superior), (x_inferior, y_inferior), (255, 0, 0), 1)
    return frame

# Obtener la ruta del directorio actual donde se encuentra el archivo de Python
ruta_actual = os.path.dirname(os.path.abspath(__file__))

# Unir la ruta actual con el nombre del video
video_ruta = os.path.join(ruta_actual, 'video.mp4')

# Capturar video desde el archivo de video
cap = cv2.VideoCapture(video_ruta)

# Crear una ventana para mostrar las imágenes
cv2.namedWindow('Reproduccion', cv2.WINDOW_NORMAL)

while cap.isOpened():
    # Capturar frame a frame
    ret, frame = cap.read()
    if ret:
        frame_con_carriles = detectar_carriles(frame)
        
        # Mostrar la imagen en la ventana 'Reproduccion'
        cv2.imshow('Reproduccion', frame_con_carriles)
        
        # Esperar aproximadamente 16 ms para lograr 60 fps
        if cv2.waitKey(1) == ord('s'): #ESTE TIEMPO DE ESPERA HABRÁ QUE AJUSTARLO SEGÚN EL TIEMPO DE PROCESAMIENTO Y DEMÁS PARA LOGAR QUE SE VEA COMO EN TIEMPO REAL
            break
    else:
        break
    
# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
