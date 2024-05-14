import cv2
import numpy as np
import os

# Contadores para contar el número de frames sin detectar líneas
frames_sin_lineas_izquierdas = 0
frames_sin_lineas_derechas = 0

# Listas para almacenar las líneas de los últimos 10 frames
ultimas_lineas_izquierdas = []
ultimas_lineas_derechas = []

def detectar_carriles(frame):
    global ultimas_lineas_izquierdas, ultimas_lineas_derechas, frames_sin_lineas_izquierdas, frames_sin_lineas_derechas
    
    # Convertir el frame a escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque gaussiano para reducir el ruido
    desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)
    
    # Detectar bordes usando el operador de Canny
    bordes = cv2.Canny(desenfoque, 50, 100)
    
    # Definir una región de interés (ROI) trapezoidal
    altura, ancho = frame.shape[:2]
    ROI_VERTICES = [(0.3, 0.75), (0.7, 0.75), (1, 0.95), (0, 0.95)]  # Definir vértices del ROI
    puntos_roi = np.array([[(ancho * x, altura * y) for x, y in ROI_VERTICES]], dtype=np.int32)
    mascara = np.zeros_like(gris)
    cv2.fillPoly(mascara, puntos_roi, 255)
    imagen_roi = cv2.bitwise_and(bordes, mascara)

    # Detectar líneas usando la transformada de Hough
    lineas = cv2.HoughLinesP(imagen_roi, 1, np.pi/180, 50, minLineLength=50, maxLineGap=100)
    
    # Separar las líneas en grupos izquierda y derecha
    lineas_izquierdas = []
    lineas_derechas = []
    if lineas is not None:  # Verificar si se detectaron líneas
        for linea in lineas:
            x1, y1, x2, y2 = linea[0]
            if (x2 - x1) != 0:    
                pendiente = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita
                if pendiente < -0.5:
                    lineas_izquierdas.append(linea[0])
                elif pendiente > 0.5:
                    lineas_derechas.append(linea[0])
    # Verificar si se detectaron líneas izquierdas y actualizar el contador
    if lineas_izquierdas:
        frames_sin_lineas_izquierdas = 0
    else:
        frames_sin_lineas_izquierdas += 1
    # Verificar si se detectaron líneas derechas y actualizar el contador
    if lineas_derechas:
        frames_sin_lineas_derechas = 0
    else:
        frames_sin_lineas_derechas += 1

    # Promediar las coordenadas de las líneas izquierda y derecha si se detectaron
    if lineas_izquierdas:
        promedio_izquierdo = np.mean(lineas_izquierdas, axis=0, dtype=np.int32)
        # Histórico de las X últimas líneas
        ultimas_lineas_izquierdas.append(promedio_izquierdo)
        if len(ultimas_lineas_izquierdas) > 3:
            ultimas_lineas_izquierdas.pop(0)  # Eliminar el primer elemento si hay más de X elementos
    if lineas_derechas:
        promedio_derecho = np.mean(lineas_derechas, axis=0, dtype=np.int32)
        # Histórico de las X últimas líneas
        ultimas_lineas_derechas.append(promedio_derecho)
        if len(ultimas_lineas_derechas) > 3:
            ultimas_lineas_derechas.pop(0)  # Eliminar el primer elemento si hay más de X elementos
    
    # Dibujar líneas solo si no han pasado X frames sin detectar líneas
    if frames_sin_lineas_izquierdas < 10 and ultimas_lineas_izquierdas:
        lineas_definitivas_izquierdas = np.mean(ultimas_lineas_izquierdas, axis=0, dtype=np.int32)
        x1, y1, x2, y2 = lineas_definitivas_izquierdas
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 5)

        # Extender las líneas solo dentro del rango de la ROI
        pendiente = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita
        if pendiente != 0:  # Evitar división por cero
            interseccion_superior = int(altura * 0.75)
            interseccion_inferior = int(altura * 0.95)
            interseccion_superior_x = int((interseccion_superior - y1) / pendiente + x1)
            interseccion_inferior_x = int((interseccion_inferior - y1) / pendiente + x1)
            cv2.line(frame, (interseccion_superior_x, interseccion_superior), (interseccion_inferior_x, interseccion_inferior), (255, 0, 255), 5)
    
    if frames_sin_lineas_derechas < 10 and ultimas_lineas_derechas:
        lineas_definitivas_derechas = np.mean(ultimas_lineas_derechas, axis=0, dtype=np.int32)
        x1, y1, x2, y2 = lineas_definitivas_derechas
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 5)

        # Extender las líneas solo dentro del rango de la ROI
        pendiente = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita
        if pendiente != 0:  # Evitar división por cero
            interseccion_superior = int(altura * 0.75)
            interseccion_inferior = int(altura * 0.95)
            interseccion_superior_x = int((interseccion_superior - y1) / pendiente + x1)
            interseccion_inferior_x = int((interseccion_inferior - y1) / pendiente + x1)
            cv2.line(frame, (interseccion_superior_x, interseccion_superior), (interseccion_inferior_x, interseccion_inferior), (255, 0, 255), 5)

    # Dibujar la zona de interés con una línea discontínua verde
    cv2.polylines(frame, [puntos_roi], isClosed=True, color=(0, 255, 0), thickness=2)
    
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
