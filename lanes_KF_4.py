import cv2
import numpy as np
import os

estado_ini = [500, 800, 1500, 1150, 0, 0]
ancho_inf_ini = estado_ini[2]-estado_ini[0]
ancho_sup_ini = estado_ini[3]-estado_ini[1]

A = np.array([[1,0,0,0,1,0],
              [0,1,0,0,0,1],
              [0,0,1,0,1,0],
              [0,0,0,1,0,1],
              [0,0,0,0,1,0],
              [0,0,0,0,0,1]])

C = np.array([[1,0,0,0,0,0],
              [0,1,0,0,0,0],
              [0,0,1,0,0,0],
              [0,0,0,1,0,0],
              [1,0,0,0,0,0],
              [0,1,0,0,0,0],
              [0,0,1,0,0,0],
              [0,0,0,1,0,0],
              [1,0,0,0,0,0],
              [0,1,0,0,0,0],
              [0,0,1,0,0,0],
              [0,0,0,1,0,0]])

Q_ini = np.diag(np.concatenate(([0.001]*4, [0.0001]*2))) # confianza en el modelo
Q_ini[(4,5)]=0.8*Q_ini[(4,5)]*Q_ini[(5,4)]
Q_ini[(5,4)]=Q_ini[(4,5)]
P_ini = np.eye(6)*0.0001
R_ini = np.eye(12)*1 # confianza en las medidas
K_ini = np.ones((6,12))*0.001

t_d=0
t_i=0

t_err_ancho=0
t_err_pos=0
t_err_lineaizq=0
t_err_lineader=0

def detectar_carriles(frame,estado,Q,P,R,K):
    global ultimas_lineas_izquierdas, ultimas_lineas_derechas, t_d, t_i
    global A, C, Q_ini, P_ini, R_ini, K_ini, ancho_inf_ini, ancho_sup_ini
    global t_err_ancho, t_err_pos, t_err_lineaizq, t_err_lineader

    altura, ancho = frame.shape[:2]

    estado_estimado = A @ estado

    x1, y1, x2, y2 = [estado_estimado[0],altura*0.95,estado_estimado[1],altura*0.75]
    pendiente_i = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita

    x1, y1, x2, y2 = [estado_estimado[2],altura*0.95,estado_estimado[3],altura*0.75]
    pendiente_d = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita

    xpunto_inf, xpunto_sup = [estado[4],estado[5]]
    if xpunto_inf > 0 and xpunto_sup > 0 and abs(pendiente_i) > 10:
        # cambio al carril de la izquierda:
        # la linea de la derecha hereda las caracteristicas de la que era la linea de la izquierda
        estado_estimado[2] = estado_estimado[0]
        estado_estimado[3] = estado_estimado[1]
        estado_estimado[0] = estado_estimado[2]-ancho_inf_ini
        estado_estimado[1] = estado_estimado[3]-ancho_sup_ini

    elif xpunto_inf < 0 and xpunto_sup < 0 and abs(pendiente_d) > 10:
        # cambio al carril de la derecha
        # la linea de la izquierda hereda las caracteristicas de la que era la linea de la derecha
        estado_estimado[0] = estado_estimado[2]
        estado_estimado[1] = estado_estimado[3]
        estado_estimado[2] = estado_estimado[0]+ancho_inf_ini
        estado_estimado[3] = estado_estimado[1]+ancho_sup_ini

    x_izquierda_inf, x_izquierda_sup, x_derecha_inf, x_derecha_sup = [estado_estimado[0],estado_estimado[1],estado_estimado[2],estado_estimado[3]]
    x_inf_mid = (x_izquierda_inf+x_derecha_inf)*0.5
    ancho_inf = x_derecha_inf - x_izquierda_inf
    ancho_sup = (x_izquierda_sup+x_derecha_sup)*0.5

    if ancho_inf > 1.25*ancho_inf_ini or abs(x_inf_mid-ancho>200):
        t_err_ancho += 1
    else:
        t_err_ancho = 0

    if abs(x_inf_mid-ancho>200):
        t_err_pos +=1
    else:
        t_err_pos = 0
    

    if t_err_ancho > 20 or t_err_pos > 20 or (t_err_lineaizq > 60 and t_err_lineader > 60):
        if t_err_ancho > 20:
            print("ancho incorrecto")
        if t_err_pos > 20:
            print("posicion incorrecta")
        if t_err_lineaizq > 20 and t_err_lineader > 20:
            print("linea izquierda incorrecta: ", t_err_lineaizq)
            print("linea derecha incorrecta: ", t_err_lineader)
        print("reiniciando...")
        estado_estimado = estado_ini
        t_err_ancho=0
        t_err_pos=0
        t_err_lineaizq=0
        t_err_lineader=0

    P = A @ P @ A.T + Q
    medida_estimada = C @ estado_estimado



    # Definición de la región de interés (ROI)
    puntos_roi = np.array([[(ancho*0.35, altura*0.75), (ancho*0.65, altura*0.75), (ancho*1, altura*0.95), (ancho*0, altura*0.95)]], dtype=np.int32)
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
                pendiente = (y2 - y1) / (x2 - x1)
                if pendiente != 0:
                    y_inf = int(altura*0.95)
                    x_inf = int((y_inf - y1) / pendiente + x1)
                    y_sup = int(altura*0.75)
                    x_sup = int((y_sup - y1) / pendiente + x1)
                    if abs(pendiente) > 0.45:
                        #cv2.line(frame, (x1, y1),(x2, y2), (0, 255, 255), 2)
                        if x_inf<=(x_izquierda_inf + ancho_inf*0.25) and x_sup<=(x_izquierda_sup + ancho_sup*0.25):
                            lineas_izquierdas.append(np.array([[x_inf, y_inf, x_sup, y_sup]]))
                        if x_inf>=(x_derecha_inf - ancho_inf*0.25) and x_sup>=(x_derecha_sup - ancho_sup*0.25):
                            lineas_derechas.append(np.array([[x_inf, y_inf, x_sup, y_sup]]))
    
    if lineas_izquierdas:
        lineas_izquierdas_promedio = np.mean(lineas_izquierdas, axis=0, dtype=np.int32)
        x_izquierda_inf_promedio, y_izquierda_inf_promedio, x_izquierda_sup_promedio, y_izquierda_sup_promedio = lineas_izquierdas_promedio[0]
        lineas_izquierdas_mediana = np.median(lineas_izquierdas, axis=0)
        x_izquierda_inf_mediana, y_izquierda_inf_mediana, x_izquierda_sup_mediana, y_izquierda_sup_mediana = lineas_izquierdas_mediana[0]
        em_min = float('inf') # error medio con respecto a la linea predicha por el modelo
        for linea in lineas_izquierdas:
            x_inf,y_inf,x_sup,y_sup = linea[0]
            e_inf = x_inf - estado_estimado[0] # error entre la x inferior de la linea y la x inferior predicha por el modelo
            e_sup = x_sup - estado_estimado[1] # error entre la x superior de la linea y la x superior predicha por el modelo
            if (e_inf >= 0 and e_sup >= 0) or (e_inf < 0 and e_sup < 0):
                em = abs((e_inf)+(e_sup)/2*(y_inf-y_sup)) # integral del error
            else:
                y_interseccion = y_inf + (y_sup-y_inf)*abs(e_inf)/(abs(e_inf)+abs(e_sup))
                em = abs(e_inf)*(y_inf-y_interseccion)/2 + abs(e_sup)*(y_interseccion-y_sup)/2 # integral del error
            if em < em_min:
                em_min = em
                lineas_izquierdas_em_min = linea
                x_izquierda_inf_em_min, y_izquierda_inf_em_min, x_izquierda_sup_em_min, y_izquierda_sup_em_min = lineas_izquierdas_em_min[0]
        t_err_lineaizq = 0
    else:
        x_izquierda_inf_promedio, y_izquierda_inf_promedio, x_izquierda_sup_promedio, y_izquierda_sup_promedio = [estado_estimado[0],altura*0.95,estado_estimado[1],altura*0.75]
        x_izquierda_inf_mediana, y_izquierda_inf_mediana, x_izquierda_sup_mediana, y_izquierda_sup_mediana = [estado_estimado[0],altura*0.95,estado_estimado[1],altura*0.75]
        x_izquierda_inf_em_min, y_izquierda_inf_em_min, x_izquierda_sup_em_min, y_izquierda_sup_em_min = [estado_estimado[0],altura*0.95,estado_estimado[1],altura*0.75]
        t_err_lineaizq +=1

    if lineas_derechas:
        lineas_derechas_promedio = np.mean(lineas_derechas, axis=0, dtype=np.int32)
        x_derecha_inf_promedio, y_derecha_inf_promedio, x_derecha_sup_promedio, y_derecha_sup_promedio = lineas_derechas_promedio[0]
        lineas_derechas_mediana = np.median(lineas_derechas, axis=0)
        x_derecha_inf_mediana, y_derecha_inf_mediana, x_derecha_sup_mediana, y_derecha_sup_mediana = lineas_derechas_mediana[0]
        em_min = float('inf') # error medio con respecto a la linea predicha por el modelo
        for linea in lineas_derechas:
            x_inf,y_inf,x_sup,y_sup = linea[0]
            e_inf = x_inf - estado_estimado[2] # error entre la x inferior de la linea y la x inferior predicha por el modelo
            e_sup = x_sup - estado_estimado[3] # error entre la x superior de la linea y la x superior predicha por el modelo
            if (e_inf >= 0 and e_sup >= 0) or (e_inf < 0 and e_sup < 0):
                em = abs((e_inf)+(e_sup)/2*(y_inf-y_sup)) # integral del error
            else:
                y_interseccion = y_inf + (y_sup-y_inf)*abs(e_inf)/(abs(e_inf)+abs(e_sup))
                em = abs(e_inf)*(y_inf-y_interseccion)/2 + abs(e_sup)*(y_interseccion-y_sup)/2 # integral del error
            if em < em_min:
                em_min = em
                lineas_derechas_em_min = linea
                x_derecha_inf_em_min, y_derecha_inf_em_min, x_derecha_sup_em_min, y_derecha_sup_em_min = lineas_derechas_em_min[0]
        t_err_lineader = 0
    else:
        x_derecha_inf_promedio, y_derecha_inf_promedio, x_derecha_sup_promedio, y_derecha_sup_promedio = [estado_estimado[2],altura*0.95,estado_estimado[3],altura*0.75]
        x_derecha_inf_mediana, y_derecha_inf_mediana, x_derecha_sup_mediana, y_derecha_sup_mediana = [estado_estimado[2],altura*0.95,estado_estimado[3],altura*0.75]
        x_derecha_inf_em_min, y_derecha_inf_em_min, x_derecha_sup_em_min, y_derecha_sup_em_min = [estado_estimado[2],altura*0.95,estado_estimado[3],altura*0.75]
        t_err_lineader += 1

    medida = [
        x_izquierda_inf_promedio,x_izquierda_sup_promedio,x_derecha_inf_promedio,x_derecha_sup_promedio,
        x_izquierda_inf_mediana,x_izquierda_sup_mediana,x_derecha_inf_mediana,x_derecha_sup_mediana,
        x_izquierda_inf_em_min,x_izquierda_sup_em_min,x_derecha_inf_em_min,x_derecha_sup_em_min
        ]
    
    estado = estado_estimado + K @ (medida - medida_estimada)
    P = P - K @ C @ P
    K = P @ C.T @ (C @ P @ C.T + R).T

    x1, y1, x2, y2 = [estado[0],altura*0.95,estado[1],altura*0.75]
    # Extender las líneas solo dentro del rango de la ROI
    pendiente_i = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita
    if pendiente_i != 0:  # Evitar división por cero
        y_izquierda_inf = int(altura*0.95)
        x_izquierda_inf = int((y_izquierda_inf - y1) / pendiente_i + x1)
        y_izquierda_sup = int(altura*0.75)
        x_izquierda_sup = int((y_izquierda_sup - y1) / pendiente_i + x1)
        cv2.line(frame, (x_izquierda_inf, y_izquierda_inf),(x_izquierda_sup, y_izquierda_sup), (255, 0, 255), 5)

    x1, y1, x2, y2 = [estado[2],altura*0.95,estado[3],altura*0.75]
    pendiente_d = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Manejar caso de pendiente infinita
    if pendiente_d != 0:  # Evitar división por cero
        y_derecha_inf = int(altura*0.95)
        x_derecha_inf = int((y_derecha_inf - y1) / pendiente_d + x1)
        y_derecha_sup = int(altura*0.75)
        x_derecha_sup = int((y_derecha_sup - y1) / pendiente_d + x1)
        cv2.line(frame, (x_derecha_inf, y_derecha_inf),(x_derecha_sup, y_derecha_sup), (255, 0, 255), 5)


    x_sup_mid = int((x_izquierda_sup+x_derecha_sup)*0.5)
    x_inf_mid = int((x_izquierda_inf+x_derecha_inf)*0.5)
    #cv2.line(frame, (x_sup_mid, y_derecha_sup), (x_inf_mid, y_derecha_inf), (255, 0, 255), 2)
    
    if (x_inf_mid-ancho/2)>250 and t_i>=t_d:
        start_point = (ancho//3, 1032)
        end_point = (2*ancho//3, 1032)
        color = (0, 0, 255)
        thickness = 25
        cv2.arrowedLine(frame, start_point, end_point, color, thickness)

        frame_cpy = frame.copy()
        cv2.fillPoly(frame, np.array([[(x_izquierda_sup, y_izquierda_sup), (x_izquierda_inf, y_izquierda_inf), (x_derecha_inf, y_derecha_inf), (x_derecha_sup, y_derecha_sup)]], dtype=np.int32), color=(0, 0, 255))
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
        cv2.fillPoly(frame, np.array([[(x_izquierda_sup, y_izquierda_sup), (x_izquierda_inf, y_izquierda_inf), (x_derecha_inf, y_derecha_inf), (x_derecha_sup, y_derecha_sup)]], dtype=np.int32), color=(0, 0, 255))
        alpha = 0.4
        frame = cv2.addWeighted(frame, alpha, frame_cpy, 1 - alpha, gamma=0)

        t_d+=1

    else:
        frame_cpy = frame.copy()
        cv2.fillPoly(frame, np.array([[(x_izquierda_sup, y_izquierda_sup), (x_izquierda_inf, y_izquierda_inf), (x_derecha_inf, y_derecha_inf), (x_derecha_sup, y_derecha_sup)]], dtype=np.int32), color=(255, 0, 255))
        alpha = 0.4
        frame = cv2.addWeighted(frame, alpha, frame_cpy, 1 - alpha, gamma=0)

        if t_d>0:
            t_d-=1
        if t_i>0:
            t_i-=1

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

estado = estado_ini

Q = Q_ini
P = P_ini
R = R_ini
K = K_ini

while cap.isOpened():
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
