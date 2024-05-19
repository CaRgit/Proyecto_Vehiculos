import cv2
import numpy as np
import os

ultimas_lineas_izquierdas = []
ultimas_lineas_derechas = []

A = np.array([
    [1,0,0,0,1,0],
    [0,1,0,0,0,1],
    [0,0,1,0,1,0],
    [0,0,0,1,0,1],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1]
])

C = np.array([
    [1,0,0,0,0,0],
    [0,1,0,0,0,0],
    [0,0,1,0,0,0],
    [0,0,0,1,0,0]
    ])

t_d=0
t_i=0

def detectar_carriles(frame,estado,Q,P,R,K):
    global ultimas_lineas_izquierdas, ultimas_lineas_derechas, t_d, t_i
    global A, C

    estado_estimado = A @ estado
    P = A @ P @ A.T + Q
    medida_estimada = C @ estado_estimado

    # Definición de la región de interés (ROI)
    altura, ancho = frame.shape[:2]
    puntos_roi = np.array([[(0.35*ancho, 0.75*altura), (0.65*ancho, 0.75*altura), (ancho*1, altura*0.95), (0*ancho, altura*0.95)]], dtype=np.int32)
    mascara = np.zeros_like(frame)
    cv2.fillPoly(mascara, puntos_roi, (255, 255, 255))
    frame_recortado = cv2.bitwise_and(frame, mascara)
    imagen_roi = cv2.resize(frame_recortado, (960, 540)) # Redimensionamiento de la ROI
    #cv2.imshow('Imagen', imagen_roi)
  
    # Convertir imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen_roi, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Imagen', imagen_gris)
    
    # Aplicar desenfoque gaussiano para reducir el ruido
    imagen_desenfocada = cv2.GaussianBlur(imagen_gris, (5, 5), 0)
    #cv2.imshow('Imagen', imagen_desenfocada)
    
    # Aplicación de ecualización del histograma
    imagen_ecualizada_i=cv2.equalizeHist(imagen_desenfocada[:,:960//2])
    imagen_ecualizada_d=cv2.equalizeHist(imagen_desenfocada[:,960//2:])
    imagen_ecualizada=cv2.hconcat([imagen_ecualizada_i, imagen_ecualizada_d])
    #cv2.imshow('Imagen', imagen_ecualizada)

    # Función cúbica
    imagen_float_i = np.float32(imagen_ecualizada_i)
    f_cubica_i = (imagen_float_i ** 3) / (255 ** 2)
    imagen_f_cubica_i=cv2.normalize(f_cubica_i, None, 0, 255, cv2.NORM_MINMAX)
    imagen_f_cubica_i= np.uint8(imagen_f_cubica_i)

    imagen_float_d = np.float32(imagen_ecualizada_d)
    f_cubica_d = (imagen_float_d ** 3) / (255 ** 2)
    imagen_f_cubica_d=cv2.normalize(f_cubica_d, None, 0, 255, cv2.NORM_MINMAX)
    imagen_f_cubica_d= np.uint8(imagen_f_cubica_d)

    # Encuentra los puntos mínimo y máximo del histograma
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imagen_f_cubica_i)
    umbral1 = (max_val + np.mean(imagen_f_cubica_i)) / 2
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imagen_f_cubica_d)
    umbral2 = (max_val + np.mean(imagen_f_cubica_d)) / 2

    # Aplica un umbral por separado a la mitad izquierda y derecha
    _, imagen_f_cubica_i = cv2.threshold(imagen_f_cubica_i, umbral1, 255, cv2.THRESH_BINARY)
    _, imagen_f_cubica_d = cv2.threshold(imagen_f_cubica_d, umbral2, 255, cv2.THRESH_BINARY)
    # Une las imagenes umbralizadas
    imagen_umbralizada = cv2.hconcat([imagen_f_cubica_i, imagen_f_cubica_d])
    # Aplica una operación morfológica de apertura para eliminar los puntos blancos pequeños
    imagen_umbralizada = cv2.morphologyEx(imagen_umbralizada, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))
    #cv2.imshow('Imagen', imagen_umbralizada)
    
    # Detectar bordes usando el operador de Canny
    imagen_bordes = cv2.Canny(cv2.resize(imagen_umbralizada, (ancho, altura)), 50, 100)
    #cv2.imshow('Imagen', imagen_bordes)

    # Detectar líneas usando la transformada de Hough
    lineas = cv2.HoughLinesP(imagen_bordes, 1, np.pi/180, 50, minLineLength=50, maxLineGap=100)


    # Separar las líneas en grupos izquierda y derecha
    lineas_izquierdas = []
    lineas_derechas = []
 
    if lineas is not None:
        for linea in lineas:
            x1, y1, x2, y2 = linea[0]
            if (x2 - x1) != 0:
                pendiente = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf') 
                if (pendiente < -0.45 and x1<=ancho/2-200) or np.isinf(pendiente):
                    lineas_izquierdas.append(linea[0])
                if (pendiente > 0.45 and x1>=ancho/2+200) or np.isinf(pendiente):
                    lineas_derechas.append(linea[0])    
    
    
    if lineas_izquierdas:
        promedio_izquierdo = np.mean(lineas_izquierdas, axis=0, dtype=np.int32)
        ultimas_lineas_izquierdas.append(promedio_izquierdo)
        if len(ultimas_lineas_izquierdas) > 1:
            ultimas_lineas_izquierdas.pop(0)
        lineas_definitivas_izquierdas = np.mean(ultimas_lineas_izquierdas, axis=0, dtype=np.int32)
        x1, y1, x2, y2 = lineas_definitivas_izquierdas
    else:
        x1, y1, x2, y2 = [estado_estimado[0],altura,estado_estimado[1],0]
    pendiente_i = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita
    if pendiente_i != 0:
        x_izquierda_inf = int((altura - y1) / pendiente_i + x1)
        x_izquierda_sup = int((0 - y1) / pendiente_i + x1)

    if lineas_derechas:
        promedio_derecho = np.mean(lineas_derechas, axis=0, dtype=np.int32)
        ultimas_lineas_derechas.append(promedio_derecho)
        if len(ultimas_lineas_derechas) > 1:
            ultimas_lineas_derechas.pop(0) 

        lineas_definitivas_derechas = np.mean(ultimas_lineas_derechas, axis=0, dtype=np.int32)
        x1, y1, x2, y2 = lineas_definitivas_derechas
    else:
        x1, y1, x2, y2 = [estado_estimado[2],altura,estado_estimado[3],0]
    pendiente_d = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita
    if pendiente_d != 0:
        x_derecha_inf = int((altura - y1) / pendiente_d + x1)
        x_derecha_sup = int((0 - y1) / pendiente_d + x1)

    medida = [x_izquierda_inf, x_izquierda_sup, x_derecha_inf, x_derecha_sup]
    estado = estado_estimado + K @ (medida - medida_estimada)
    P = P - K @ C @ P
    K = P @ C.T @ (C @ P @ C.T + R).T

    x1, y1, x2, y2 = [estado[0],altura,estado[1],0]
    # Extender las líneas solo dentro del rango de la ROI
    pendiente_i = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita
    if pendiente_i != 0:  # Evitar división por cero
        ydraw_izquierda_inf = int(altura * 0.95)
        xdraw_izquierda_inf = int((ydraw_izquierda_inf - y1) / pendiente_i + x1)
        ydraw_izquierda_sup = int(altura * 0.75)
        xdraw_izquierda_sup = int((ydraw_izquierda_sup - y1) / pendiente_i + x1)
        cv2.line(frame, (xdraw_izquierda_inf, ydraw_izquierda_inf),(xdraw_izquierda_sup, ydraw_izquierda_sup), (255, 0, 255), 5)

    x1, y1, x2, y2 = [estado[2],altura,estado[3],0]
    pendiente_d = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita
    if pendiente_d != 0:  # Evitar división por cero
        ydraw_derecha_inf = int(altura * 0.95)
        xdraw_derecha_inf = int((ydraw_derecha_inf - y1) / pendiente_d + x1)
        ydraw_derecha_sup = int(altura * 0.75)
        xdraw_derecha_sup = int((ydraw_derecha_sup - y1) / pendiente_d + x1)
        cv2.line(frame, (xdraw_derecha_inf, ydraw_derecha_inf),(xdraw_derecha_sup, ydraw_derecha_sup), (255, 0, 255), 5)


    x_sup_mid = int((xdraw_izquierda_sup+xdraw_derecha_sup)*0.5)
    x_inf_mid = int((xdraw_izquierda_inf+xdraw_derecha_inf)*0.5)
    cv2.line(frame, (x_sup_mid, ydraw_derecha_sup), (x_inf_mid, ydraw_derecha_inf), (255, 0, 255), 2)
    
    if (x_inf_mid-ancho/2)>250 and t_i>=t_d:
        start_point = (ancho//3, 1032)
        end_point = (2*ancho//3, 1032)
        color = (0, 0, 255)
        thickness = 25
        cv2.arrowedLine(frame, start_point, end_point, color, thickness)

        frame_cpy = frame.copy()
        cv2.fillPoly(frame, np.array([[(xdraw_izquierda_sup, ydraw_izquierda_sup), (xdraw_izquierda_inf, ydraw_izquierda_inf), (xdraw_derecha_inf, ydraw_derecha_inf), (xdraw_derecha_sup, ydraw_derecha_sup)]], dtype=np.int32), color=(0, 0, 255))
        alpha = 0.4
        frame = cv2.addWeighted(frame, alpha, frame_cpy, 1 - alpha, gamma=0)

        t_i+=1

    elif (ancho/2-x_inf_mid)>250 and t_d>=t_i:
        start_point = (2*ancho//3, 1032)
        end_point = (ancho//3, 1032)
        color = (0, 0, 255)
        thickness = 25
        cv2.arrowedLine(frame, start_point, end_point, color, thickness)

        frame_cpy = frame.copy()
        cv2.fillPoly(frame, np.array([[(xdraw_izquierda_sup, ydraw_izquierda_sup), (xdraw_izquierda_inf, ydraw_izquierda_inf), (xdraw_derecha_inf, ydraw_derecha_inf), (xdraw_derecha_sup, ydraw_derecha_sup)]], dtype=np.int32), color=(0, 0, 255))
        alpha = 0.4
        frame = cv2.addWeighted(frame, alpha, frame_cpy, 1 - alpha, gamma=0)

        t_d+=1

    else:
        frame_cpy = frame.copy()
        cv2.fillPoly(frame, np.array([[(xdraw_izquierda_sup, ydraw_izquierda_sup), (xdraw_izquierda_inf, ydraw_izquierda_inf), (xdraw_derecha_inf, ydraw_derecha_inf), (xdraw_derecha_sup, ydraw_derecha_sup)]], dtype=np.int32), color=(255, 0, 255))
        alpha = 0.4
        frame = cv2.addWeighted(frame, alpha, frame_cpy, 1 - alpha, gamma=0)

        if t_d>0:
            t_d-=1
        if t_i>0:
            t_i-=1
    
    # Dibujar la zona de interés con una línea discontínua verde
    cv2.polylines(frame, [puntos_roi], isClosed=True, color=(0, 255, 0), thickness=2)


    return frame, estado, P, K





# Obtener la ruta del directorio actual donde se encuentra el archivo de Python
ruta_actual = os.path.dirname(os.path.abspath(__file__))

# Unir la ruta actual con el nombre del video
video_ruta = os.path.join(ruta_actual, 'video_f.mp4')

# Capturar video desde el archivo de video
cap = cv2.VideoCapture(video_ruta)

# Crear una ventana para mostrar las imágenes
cv2.namedWindow('Reproduccion', cv2.WINDOW_NORMAL)

init = True
while init:
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            altura, ancho = frame.shape[:2]
            init = False

estado = [404, 1834, 1671, -583, 0, 0]
ultimas_lineas_izquierdas.append([244, altura, 2040, 0])
ultimas_lineas_derechas.append([1483, altura, -317, 0])
Q = np.eye(6)*0.00001
P = np.eye(6)*0.01
R = np.eye(4)*25
K = np.array([
    [0.001,0.001,0.001,0.001],
    [0.001,0.001,0.001,0.001],
    [0.001,0.001,0.001,0.001],
    [0.001,0.001,0.001,0.001],
    [0.001,0.001,0.001,0.001],
    [0.001,0.001,0.001,0.001]
])

while cap.isOpened():
    # Capturar frame a frame
    ret, frame = cap.read()
    if ret:
        frame_con_carriles, estado, P, K = detectar_carriles(frame,estado,Q,P,R,K)

        cv2.imshow('Reproduccion', frame_con_carriles)

        if cv2.waitKey(1) == ord('s'):  
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()