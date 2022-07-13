import tkinter
import math
import os
from statistics import mode
from tkinter.ttk import Progressbar
import pandas as pd
import pyttsx3
import numpy as np
from imutils.perspective import four_point_transform
import imutils
from skimage import filters, metrics
from matplotlib import pyplot as plt
from tkinter import messagebox as MessageBox
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
from numpy import argmax
import statistics

ix1, iy1, ix2, iy2 = -1, -1, -1, -1
led_seleccionado = False


def seleccionar_led(imagen):
    # mouse call
    def armar_mascara(event, x, y, flags, param):
        global ix1, iy1, ix2, iy2, mascara_armada, led_seleccionado
        if event == cv2.EVENT_LBUTTONDOWN:
            if ix1 == -1 and iy1 == -1:  # primera asignacion
                ix1, iy1 = x, y
                led_seleccionado = True
            else:  # otras reasignaciones
                ix2 = ix1
                iy2 = iy1
                ix1, iy1 = x, y
                led_seleccionado = True

    if not led_seleccionado:
        cv2.namedWindow('Seleccionar led')
        cv2.setMouseCallback('Seleccionar led', armar_mascara)
        while 1:
            if ix1 != -1 and iy1 != -1 and ix2 != -1 and iy2 != -1 or cv2.getWindowProperty('Seleccionar led', cv2.WND_PROP_VISIBLE) <1:  # ya armamos la mascara
                break
            cv2.imshow('Seleccionar led', imagen)
            cv2.waitKey(1)
        cv2.destroyAllWindows()

    return ix1, iy1, ix2, iy2


# It's just a text to speech function..
def saySomething(somethingToSay):
    engine = pyttsx3.init()
    engine.setProperty('rate', 120)
    engine.setProperty('voice', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0')
    engine.setProperty('volume', 40)

    engine.say(somethingToSay)
    engine.runAndWait()


def dictado_voz(tipo_led, digitos, habla):
    quedice = ""
    if int(tipo_led) == 1 or int(tipo_led) == 2:
        if len(digitos) == 5:
            digitos = sorted(digitos, key=lambda x: x[1], reverse=True)
            sisole = [digitos[2], digitos[3], digitos[4]]
            sisole = sorted(sisole, key=lambda x: x[0], reverse=False)
            sistole = str(sisole[0][2]) + str(sisole[1][2]) + str(sisole[2][2])
            diastol = [digitos[0], digitos[1]]
            diastol = sorted(diastol, key=lambda x: x[0], reverse=False)
            diastole = str(diastol[0][2]) + str(diastol[1][2])
            quedice = "La sistole es " + sistole + " y la diastole es " + diastole
        else:
            digitos = sorted(digitos, key=lambda x: x[0], reverse=False)
            quedice = ""
            for i in range(len(digitos)):
                quedice = str(quedice) + str(digitos[i][2])

    if int(tipo_led) == 3:  # balanza verde
        digitos = sorted(digitos, key=lambda x: x[0], reverse=False)
        quedice2 = ""
        for i in range(len(digitos)):
            quedice2 = str(quedice2) + str(digitos[i][2])
        if len(digitos) < 5:
            quedice = "El peso es " + quedice2[0] + "coma" + quedice2[1:] + " kilogramos"
        else:
            quedice = "El peso es " + quedice2[0] + quedice2[1] + "coma" + quedice2[2:] + " kilogramos"

    if int(tipo_led) == 4:  # balanza gris
        digitos = sorted(digitos, key=lambda x: x[0], reverse=False)
        digitos_dictar = ""
        for i in range(len(digitos)):
            digitos_dictar = str(digitos_dictar) + str(digitos[i][2])
        quedice = "El peso es " + digitos_dictar + " gramos"
    if habla:
        saySomething(quedice)

    if int(tipo_led) == 5:  # fuente azul
        digitos = sorted(digitos, key=lambda x: x[0], reverse=False)
        quedice2 = ""
        for i in range(len(digitos)):
            quedice2 = str(quedice2) + str(digitos[i][2])
        if len(digitos) == 3:
            quedice = quedice2[0] + quedice2[1] + "." + quedice2[2:]
        else:
            quedice = quedice2

    return quedice


def segmentar_imagen_hsv(imagen, h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto):
    imagen_HSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    # el color rojo esta al inicio y al final del rectangulo
    Bajo = np.array([h_bajo, s_bajo, v_bajo], np.uint8)
    Alto = np.array([h_alto, s_alto, v_alto], np.uint8)
    mascara = cv2.inRange(imagen_HSV, Bajo, Alto)
    segmentada = cv2.bitwise_and(imagen, imagen, mask=mascara)

    return segmentada, mascara


def definir_parametros(tipo_display):
    (win_size_sauvola, k_sauvo, tamanio_led, thres_canny_low, thres_canny_high, h_bajo, h_alto, s_bajo,
     s_alto, v_bajo, v_alto, tam_close, delta, min_area, max_area, max_variation, corte_display_filas_i,
     corte_display_filas_f, corte_display_columnas_i, corte_display_columnas_f, thres_blanco, porcetaje_w_display,
     porcetaje_h_display, porcetaje_w_digitos, porcetaje_h_digitos) = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                                                                      0, 0, 0, 0, 0, 0, 0, 0
    # parametros segun display
    if tipo_display == 1:  # tensiometro citizen
        porcetaje_w_display, porcetaje_h_display = 0.2, 0.4
        porcetaje_w_digitos, porcetaje_h_digitos = 0.2, 0.8
        win_size_sauvola, k_sauvo = 61, 0.2
        tamanio_led = (440, 300)

        thres_canny_low, thres_canny_high = 50, 255
        # h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto = 0, 179, 0, 106, 50, 110
        h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto = 0, 179, 10, 110, 50, 110
        # h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto = 50, 122, 0, 255, 0, 110
        tam_close = 10
        delta, min_area, max_area, max_variation = 5, 1000, 6000, 0.25
        corte_display_filas_i = 30
        corte_display_columnas_i = 20
        corte_display_filas_f = 440 - 10
        corte_display_columnas_f = -1
        thres_blanco = 150
    if tipo_display == 2:  # tensiometro gama
        porcetaje_w_display, porcetaje_h_display = 0.2, 0.3
        porcetaje_w_digitos, porcetaje_h_digitos = 0.1, 0.8
        win_size_sauvola, k_sauvo = 31, 0.2
        tamanio_led = (360, 340)
        thres_canny_low, thres_canny_high = 50, 255
        # h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto = 0, 179, 0, 106, 50, 110
        h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto = 0, 179, 0, 106, 0, 110
        tam_close = 15
        delta, min_area, max_area, max_variation = 5, 1000, 7000, 0.25
        corte_display_filas_i = 30
        corte_display_columnas_i = 0
        corte_display_filas_f = -1
        corte_display_columnas_f = -1
        thres_blanco = 150

    if tipo_display == 3:  # balanza systel brumer

        porcetaje_w_display, porcetaje_h_display = 0.2, 0.4
        porcetaje_w_digitos, porcetaje_h_digitos = 0.2, 0.8
        win_size_sauvola, k_sauvo = 61, 0.1
        tamanio_led = (220, 150)
        thres_canny_low, thres_canny_high = 0, 255
        # h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto = 0, 130, 0, 100, 200, 255
        h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto = 20, 70, 0, 100, 200, 255
        tam_close = 5
        delta, min_area, max_area, max_variation = 5, 60, 10000, 0.25
        corte_display_filas_i = 0
        corte_display_columnas_i = 0
        corte_display_filas_f = -1
        corte_display_columnas_f = -1
        thres_blanco = 150

    if tipo_display == 4:  # balanza display gris
        porcetaje_w_display, porcetaje_h_display = 0.2, 0.4
        porcetaje_w_digitos, porcetaje_h_digitos = 0.2, 0.9
        win_size_sauvola, k_sauvo = 31, 0.3
        tamanio_led = (420, 200)
        thres_canny_low, thres_canny_high = 50, 255
        h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto = 20, 100, 20, 100, 60, 150
        tam_close = 10
        delta, min_area, max_area, max_variation = 5, 2000, 8000, 0.25
        corte_display_filas_i = 5
        corte_display_columnas_i = 5
        corte_display_filas_f = -1
        corte_display_columnas_f = -1
        thres_blanco = 150

    if tipo_display == 5:  # display azul
        porcetaje_w_display, porcetaje_h_display = 0.2, 0.4
        porcetaje_w_digitos, porcetaje_h_digitos = 0.2, 0.9
        win_size_sauvola, k_sauvo = 61, 0.1
        tamanio_led = (420, 200)
        thres_canny_low, thres_canny_high = 50, 255
        h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto = 90, 115, 40, 255, 170, 255
        tam_close = 5
        delta, min_area, max_area, max_variation = 5, 1000, 8000, 0.25
        corte_display_filas_i = 5
        corte_display_columnas_i = 5
        corte_display_filas_f = -1
        corte_display_columnas_f = -1
        thres_blanco = 150
    if tipo_display == 6:  # display rojo
        porcetaje_w_display, porcetaje_h_display = 0.2, 0.4
        porcetaje_w_digitos, porcetaje_h_digitos = 0.2, 0.8
        win_size_sauvola, k_sauvo = 61, 0.1
        tamanio_led = (600, 200)
        thres_canny_low, thres_canny_high = 50, 255
        h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto = 0, 179, 102, 175, 240, 255
        tam_close = 10
        delta, min_area, max_area, max_variation = 5, 500, 3000, 0.25
        corte_display_filas_i = 0
        corte_display_columnas_i = 0
        corte_display_filas_f = -1
        corte_display_columnas_f = -1
        thres_blanco = 150
    parametros = (win_size_sauvola, k_sauvo, tamanio_led, thres_canny_low, thres_canny_high, h_bajo, h_alto, s_bajo,
                  s_alto, v_bajo, v_alto, tam_close, delta, min_area, max_area, max_variation, corte_display_filas_i,
                  corte_display_filas_f, corte_display_columnas_i, corte_display_columnas_f,
                  thres_blanco, porcetaje_w_display,
                  porcetaje_h_display, porcetaje_w_digitos, porcetaje_h_digitos)
    return parametros


def pre_proceso(imagen, h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto, tipo_led):
    # resize
    imagen_resize = imutils.resize(imagen, height=500)
    # pasamos a gris
    imagen_resize_gray = cv2.cvtColor(imagen_resize, cv2.COLOR_BGR2GRAY)
    # segmentacion
    imagen_segmentada_led, mask = segmentar_imagen_hsv(imagen_resize, h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto)
    # plt.imshow(imagen_segmentada_led,cmap='brg_r')
    # plt.show()
    if int(tipo_led) == 3 or int(tipo_led) == 5 or int(
            tipo_led) == 6:  # aplicamos moprh close para cerrar algunos huecos en caso de que no segmente del todo bien
        kernel_alto = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
        mask_dilatada_alto = cv2.dilate(mask, kernel_alto, iterations=10)
        # plt.imshow(mask_dilatada_alto,cmap='Greys_r')
        # plt.show()
        imagen_segmentada_led = cv2.bitwise_and(mask_dilatada_alto, imagen_resize_gray)
    else:  # si es retroiluminado le aplicamos una morfologia y un bitwise and
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        imagen_segmentada_led = cv2.morphologyEx(imagen_segmentada_led, cv2.MORPH_CLOSE, kernel)
    # plt.imshow(imagen_segmentada_led,cmap='Greys_r')
    # plt.show()
    # retornamos imagen_resize,imagen_segmentada_led,imagen_resize_gray para futuros usos

    return imagen_resize, imagen_segmentada_led, imagen_resize_gray, mask


def aislar_led(imagen_segmentada_led, imagen_resize_gray, imagen_resize, tamanio_led, thres_canny_low, thres_canny_high,
               corte_display_filas_i, corte_display_filas_f, corte_display_columnas_i, corte_display_columnas_f,
               tipo_led, seleccion_manual):
    # le hacemos un blureado muy pequeño y detectamos borde con canny a la imagen segmentada en gris
    if int(tipo_led) == 3 or int(tipo_led) == 5 or int(tipo_led) == 6:
        imagen_gris_segmentada = imagen_segmentada_led
    else:
        imagen_gris_segmentada = cv2.cvtColor(imagen_segmentada_led, cv2.COLOR_BGR2GRAY)
    # imagen_gris_blurred = cv2.GaussianBlur(imagen_gris_segmentada, (1, 1), 0)
    bordes = cv2.Canny(imagen_gris_segmentada, thres_canny_low, thres_canny_high, 255)
    # le hago una dilatacion a canny para cerra el area mayor
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bordes = cv2.dilate(bordes, kernel)
    if seleccion_manual == "n":
        # plt.imshow(bordes,cmap='Greys_r')
        # plt.show()
        # Hallamos los contornos  y despues se ordenan segun tamñana descendente
        contornos, _ = cv2.findContours(bordes.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
        displayCnt = None
        # Luego el que tenga 4 puntos es posible que sea el led
        # Iteramos sobre los contornos

        for c in contornos:
            # aproximamos el contorno
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            # si tiene 4 vertices encontramos el display
            if len(approx) == 4:
                displayCnt = approx
                break
        # lo extraemos al display con four_point_transform
        if displayCnt is None:
            displayCnt = [[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]]
            displayCnt = np.array(displayCnt)

        led_gris = four_point_transform(imagen_resize_gray, displayCnt.reshape(4, 2))
        led = four_point_transform(imagen_resize, displayCnt.reshape(4, 2))
    else:
        x, y, x2, y2 = seleccionar_led(imagen_resize)
        if y2 > y and x2 > x:
            led_gris = imagen_resize_gray[y:y2, x:x2]
            led = imagen_resize[y:y2, x:x2]
        if y > y2 and x > x2:
            led_gris = imagen_resize_gray[y2:y, x2:x]
            led = imagen_resize[y2:y, x2:x]
        if y > y2 and x2 > x:
            led_gris = imagen_resize_gray[y:y2, x:x2]
            led = imagen_resize[y:y2, x:x2]
        if y2 > y and x > x2:
            led_gris = imagen_resize_gray[y:y2, x:x2]
            led = imagen_resize[y:y2, x:x2]

    # lo resizeamos para mantener la relacion aspecto que se busca en los digitos (h y w)
    led = cv2.resize(led, tamanio_led)
    led_gris = cv2.resize(led_gris, tamanio_led)
    # lo recortamos por si en el led aparece algo indeceado, por ejemplo en los tensiometros aparece una barra que
    # confunde al mser
    led = led[corte_display_columnas_i:corte_display_columnas_f, corte_display_filas_i:corte_display_filas_f]
    led_gris = led_gris[corte_display_columnas_i:corte_display_columnas_f, corte_display_filas_i:corte_display_filas_f]
    # calculamos gamma semiautomatico
    mid = 0.5
    mean = np.mean(led_gris)
    if mean > 0:
        gamma = math.log(mid * 255) / math.log(mean)
    else:
        gamma = 1
    # aplicamos gamma
    led_gris = np.power(led_gris, gamma).clip(0, 255).astype(np.uint8)
    # plt.imshow(led_gris, cmap='Greys_r')
    # plt.show()

    # plt.imshow(led, cmap='brg_r')
    # plt.show()
    # retornamos led, led gris y los bordes
    return bordes, led, led_gris


def umbralizar_cerrar(led_gris, win_size_sauvola, k_sauvo, thres_blanco, tam_close, tipo_led, h_bajo, h_alto, s_bajo,
                      s_alto, v_bajo, v_alto, led):
    if int(tipo_led) == 3 or int(tipo_led) == 5 or int(tipo_led) == 6:
        _, mask = segmentar_imagen_hsv(led, h_bajo, h_alto, s_bajo, s_alto, v_bajo, v_alto)
        imagen_thresh_close, imagen_thresh = mask, mask
    else:
        # ubralizamos
        imagen_thresh = filters.threshold_sauvola(led_gris, window_size=win_size_sauvola, k=k_sauvo)
        imagen_thresh = led_gris < imagen_thresh
        imagen_thresh = np.where(imagen_thresh == False, imagen_thresh, 255)
        # el trhes queda muy blanco hacer el inverso para numeros
        if np.average(imagen_thresh) > thres_blanco:
            imagen_thresh = led_gris > imagen_thresh
            imagen_thresh = np.where(imagen_thresh == False, imagen_thresh, 255)
        imagen_thresh = imagen_thresh.astype(np.uint8)
        # hacemos un close para rellenar lo que el umbralizado no detecto, es para tener los digitos completos
        # plt.imshow(imagen_thresh, cmap='Greys_r')
        # plt.show()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tam_close, tam_close))
        imagen_thresh_close = cv2.morphologyEx(imagen_thresh, cv2.MORPH_CLOSE, kernel)
        # plt.imshow(imagen_thresh_close, cmap='Greys_r')
        # plt.show()
        # imagen_thresh_dilate=cv2.dilate(imagen_thresh,kernel,iterations=2)

    return imagen_thresh_close, imagen_thresh


def mser_detector_digitos(imagen_thresh_dilatada, delta, min_area, max_area, max_variation, tamanio_led,
                          porcet_w_display, porcet_h_display, led):
    # #Usamos MSER para detectar texto
    mser = cv2.MSER_create(delta, min_area, max_area, max_variation)
    regions, boxes = mser.detectRegions(imagen_thresh_dilatada)

    copia_led = led.copy()
    copia_led1 = led.copy()
    if len(boxes) != 0:
        idex = 0
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(copia_led, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if tamanio_led[0] * porcet_w_display < w and tamanio_led[1] * porcet_h_display < h and w * h < min_area:
                np.delete(boxes, idex)
                cv2.rectangle(copia_led1, (x, y), (x + w, y + h), (255, 0, 0), 2)
            idex = idex + 1

        boxes_sort = boxes[np.array(-boxes[:, 2]).argsort()]
        maxw = boxes_sort[0, 2]
        boxes_sort = boxes[np.array(-boxes[:, 3]).argsort()]
        maxh = boxes_sort[0, 3]
    else:
        boxes_sort = boxes
        maxw, maxh = 0, 0

    # plt.imshow(copia_led,cmap='brg_r')
    # plt.show()
    # plt.imshow(copia_led1,cmap='brg_r')
    # plt.show()
    return regions, boxes_sort, maxw, maxh


def analizar_segmentos(w, h, roi):
    # Es para el metodo de ver segmento a segmento cual esta prendido
    diccionario_de_digitos = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (1, 0, 1, 1, 1, 0, 1): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }
    # print(w / h)
    if w / h > 0.5:
        (dW, dH) = (int(w * 0.25), int(h * 0.15))
        dHC = int(h * 0.05)
        # definimos los 7 segmentos
        segments = [
            ((0, 0), (w, dH)),  # arriba centro
            ((0, 0), (dW, h // 2)),  # arriba izq
            ((w - dW, 0), (w, h // 2)),  # arriba derec
            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # seg del centro
            ((0, h // 2), (dW, h)),  # abajo izq
            ((w - dW, h // 2), (w, h)),  # abajo derecha
            ((0, h - dH), (w, h))  # abajo centro
        ]
        on = [0] * len(segments)
        # vemos los segmentos
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            segROI = roi[yA:yB, xA:xB]
            # si esta prendido
            if np.average(segROI) > 120:
                on[i] = 1
        # nos fijamos si esta en el diccionario
        try:
            digito = diccionario_de_digitos[tuple(on)]
        except:
            digito = "ND"
    else:
        digito = 1
    return digito


idex = 0

def clasificar_digitos(imagen_thresh, led, boxes, maxw, maxh, porcetaje_w_digitos, porcetaje_h_digitos):
    global idex
    imagenes_base_dir = "Base de digitos"
    files_name = os.listdir(imagenes_base_dir)
    # inicializamos las listas
    digitos = []
    # recorremos los box q encuentre y si son mayores a cierto umbral es porque son digitos
    # for box in boxes:
    for area in range(len(boxes)):
        votacion = []
        # para quedarnos con los box que queremos, es para no depender solamente del area del box
        if boxes.shape[0] > area:
            x, y, w, h = boxes[area]
            if h > maxh * porcetaje_h_digitos and w > maxw * porcetaje_w_digitos:
                # if w >= minw and (h >= minh and h <= maxh):
                # if True:
                roi = imagen_thresh[y:y + h, x:x + w]
                # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
                # roi=cv2.dilate(roi,kernel)
                # plt.imshow(roi, cmap='Greys_r')
                # plt.show()
                digito_MSE, digito_ssim, c = 0, 0, 0
                min_MSE, max_ssim = 10000000, 0
                for file_name in files_name:
                    imagen_dir = imagenes_base_dir + "/" + file_name
                    imagen_base = cv2.imread(imagen_dir)
                    if imagen_base is None:
                        continue
                    imagen_base_gris = cv2.cvtColor(imagen_base, cv2.COLOR_BGR2GRAY)
                    imagen_base_gris_resize = cv2.resize(imagen_base_gris, (roi.shape[1], roi.shape[0]))
                    MSE = metrics.mean_squared_error(imagen_base_gris_resize, roi)
                    if roi.shape[1] > 7 and roi.shape[0] > 7 and imagen_base_gris_resize.shape[1] > 7 and \
                            imagen_base_gris_resize.shape[
                                0] > 7:  # para que sea siempre mayor a 7x7 porque ssim tira error
                        SSIM = metrics.structural_similarity(imagen_base_gris_resize, roi)
                    else:
                        SSIM = 0  # inabilitamos el voto del ssim, para salvar errores
                    if MSE < min_MSE:
                        digito_MSE = c
                        min_MSE = MSE
                    if SSIM > max_ssim:
                        digito_ssim = c
                        max_ssim = SSIM
                    c = c + 1

                digito_seg = analizar_segmentos(w, h, roi)
                # el mas votado
                votacion.append(digito_MSE)
                votacion.append(digito_ssim)
                votacion.append(digito_seg)
                # digito = statistics.multimode(votacion)
                digito = mode(votacion)
                digitos.append((x, y, digito))
                cv2.rectangle(led, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(led, str(digito), (x + int(w / 2), y + int(h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 255),
                            2)
            #
            idex = idex + 1

    return digitos


def detectar_digitos(imagen, parametros, tipo_led, seleccion_manual):
    # parametros de los display
    (win_size_sauvola, k_sauvo, tamanio_led, thres_canny_low, thres_canny_high, h_bajo, h_alto, s_bajo,
     s_alto, v_bajo, v_alto, tam_close, delta, min_area, max_area, max_variation, corte_display_filas_i,
     corte_display_filas_f
     , corte_display_columnas_i, corte_display_columnas_f,
     thres_blanco, porcetaje_w_display,
     porcetaje_h_display, porcetaje_w_digitos, porcetaje_h_digitos) = parametros

    # El pre proceso es redimensionar la imagen , pasarla a escala de grises y segmentar color
    imagen_resize, imagen_segmentada_led, imagen_resize_gray, mask = pre_proceso(imagen, h_bajo, h_alto, s_bajo, s_alto,
                                                                                 v_bajo, v_alto, tipo_led)
    # detectamos los bordes y aislamos led
    bordes, led, led_gris = aislar_led(imagen_segmentada_led, imagen_resize_gray, imagen_resize, tamanio_led,
                                       thres_canny_low, thres_canny_high, corte_display_filas_i,
                                       corte_display_filas_f, corte_display_columnas_i, corte_display_columnas_f,
                                       tipo_led, seleccion_manual)
    # umbralizamos y dilatamos
    imagen_thresh_cerrada, imagen_thresh = umbralizar_cerrar(led_gris, win_size_sauvola, k_sauvo, thres_blanco,
                                                             tam_close, tipo_led, h_bajo, h_alto, s_bajo, s_alto,
                                                             v_bajo, v_alto, led)
    # plt.imshow(imagen_thresh_cerrada,cmap='Greys_r')
    # plt.show()
    # usamos mser para detectar los boxes donde pueden existir numeros
    regions, boxes, maxw, maxh = mser_detector_digitos(imagen_thresh_cerrada, delta, min_area, max_area,
                                                       max_variation, tamanio_led, porcetaje_w_display,
                                                       porcetaje_h_display, led)
    digitos = clasificar_digitos(imagen_thresh_cerrada, led, boxes, maxw, maxh, porcetaje_w_digitos,
                                 porcetaje_h_digitos)

    return digitos, imagen_resize, led, imagen_segmentada_led, imagen_thresh_cerrada, boxes


def guardar_datos():
    global quedice_list
    df = pd.DataFrame(data=quedice_list)
    df.to_excel("datos_guardados.xlsx")


seleccion_manual = "n"


def selec_manual():
    global seleccion_manual
    if var.get() == 1:
        seleccion_manual = "y"
    else:
        seleccion_manual = "n"


def hablar():
    dictado_voz(tipo_led, digitos, True)


parametros = definir_parametros(0)
tipo_led = 0


def def_param():
    global parametros, tipo_led
    if value_inside.get() == "tensiometro citizen":
        tipo_led = 1
    if value_inside.get() == "tensiometro gama muñeca":
        tipo_led = 2
    if value_inside.get() == "balanza systel retoiluminada":
        tipo_led = 3
    if value_inside.get() == "balanza led gris":
        tipo_led = 4
    if value_inside.get() == "fuente led azul":
        tipo_led = 5
    if value_inside.get() == "multímetro led rojo":
        tipo_led = 6
    parametros = definir_parametros(tipo_led)


def elegir_visualizar_video():
    global cap, ix1, iy1, ix2, iy2, led_seleccionado, quedice_list, actual_frame
    quedice_list = []
    ix1, iy1, ix2, iy2 = -1, -1, -1, -1
    led_seleccionado = False
    actual_frame = 0
    if cap is not None:
        lblVideo.image = ""
        cap.release()
        cap = None
    filety = [('all files', '.*'),
              ("all video format", ".mp4"),
              ("all video format", ".avi"),
              ('text files', '.txt'),
              ('image files', '.png'),
              ('image files', '.jpg'),
              ]
    video_path = filedialog.askopenfilename(filetypes=filety)
    if len(video_path) > 0:
        lblInfoVideoPath.configure(text=video_path)
        cap = cv2.VideoCapture(video_path)
        visualizar()
    else:
        lblInfoVideoPath.configure(text="Aún no se ha seleccionado un video")
        # lblInfoVideoPath.configure(text=video_path)
        # cap = cv2.VideoCapture(0)
        # visualizar()


quedice_list = []


def visualizar():
    global cap, frame, digitos, quedice, quedice_list, length_frame, actual_frame
    if parametros == definir_parametros(0):
        MessageBox.showwarning("Alerta", "Seleccione un tipo de LED.")
    else:
        if cap is not None:
            ret, frame = cap.read()
            length_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if ret == True:
                digitos, imagen_resize, led, imagen_segmentada_led, imagen_thresh_cerrada, boxes = detectar_digitos(
                    frame,
                    parametros,
                    tipo_led,
                    seleccion_manual)
                actual_frame = actual_frame + 1

                led = cv2.cvtColor(led, cv2.COLOR_BGR2RGB)
                quedice = dictado_voz(tipo_led, digitos, False)
                im = Image.fromarray(led)
                img = ImageTk.PhotoImage(image=im)
                lblVideo.configure(image=img)
                lblVideo.image = img
                lblVideo.after(10, visualizar)
                progressbar["value"] = actual_frame
                progressbar["maximum"] = length_frame
                quedice = dictado_voz(tipo_led, digitos, False)
                quedice_list.append(quedice)


cap = None
root = Tk()
btnVisualizar = Button(root, text="Elegir archivo", command=elegir_visualizar_video)
btnVisualizar.grid(column=0, row=0, padx=5, pady=5, columnspan=2)
lblInfo1 = Label(root, text="Video de entrada:")
lblInfo1.grid(column=0, row=1)
lblInfoVideoPath = Label(root, text="Aún no se ha seleccionado un video")
lblInfoVideoPath.grid(column=1, row=1)
lblVideo = Label(root)
lblVideo.grid(column=0, row=2, columnspan=2)
# Create the list of options
options_list = ["tensiometro citizen", "tensiometro gama muñeca", "balanza systel retoiluminada", "balanza led "
                                                                                                  "gris",
                "fuente led azul", "multímetro led rojo"]

# Variable to keep track of the option
# selected in OptionMenu
value_inside = tkinter.StringVar(root)

# Set the default value of the variable
value_inside.set("Selecciona un tipo de led")

# Create the optionmenu widget and passing
# the options_list and value_inside to it.
question_menu = tkinter.OptionMenu(root, value_inside, *options_list)
question_menu.grid()

submit_button = tkinter.Button(root, text='Definir parametros', command=def_param)
submit_button.grid()

var = IntVar()
chk = Checkbutton(root, text='Seleccion manual', variable=var, command=selec_manual)
chk.grid()

submit_button = tkinter.Button(root, text='Dictar', command=hablar)
submit_button.grid(row=3, column=2)

submit_button = tkinter.Button(root, text='Guardar datos ', command=guardar_datos)
submit_button.grid(row=3, column=1)

progressbar = Progressbar(root, orient="horizontal", length=200, mode="determinate")
progressbar.grid(row=4, column=1)

root.mainloop()
