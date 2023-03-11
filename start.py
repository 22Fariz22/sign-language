# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

#fonts
# font                   = cv2.FONT_HERSHEY_SIMPLEX
# bottomLeftCornerOfText = (100,100)
# fontScale              = 3
# fontColor              = (255,255,255)
# thickness              = 2
# lineType               = 2

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                dots = list(enumerate(hand_landmarks.landmark)) # список координатов каждой точки
                x_big = dots[4][1].x * w
                y_big = dots[4][1].y * h
                # указательный палец
                x_forefinger = dots[8][1].x * w
                y_forefinger = dots[8][1].y * h
                # средний палец
                x_middle = dots[12][1].x * w
                y_middle = dots[12][1].y * h
                # безымянный палец
                x_ring = dots[16][1].x * w
                y_ring = dots[16][1].y * h
                # маленький палец
                x_little = dots[20][1].x * w
                y_little = dots[20][1].y * h
                # запястье
                x_wrist = dots[0][1].x * w
                y_wrist= dots[0][1].y * h

                # основание большого пальца
                x_big_base = dots[1][1].x*w
                y_big_base = dots[1][1].y*w
                # основание указательного пальца
                x_forefinger_base = dots[5][1].x*w
                y_forefinger_base = dots[5][1].y*h
                # основание среднего пальца
                x_middle_base = dots[9][1].x*w
                y_middle_base = dots[9][1].y*h
                # основание безымянного пальца
                x_ring_base = dots[13][1].x*w
                y_ring_base = dots[13][1].y*h
                # основание маленького пальца
                x_little_base = dots[17][1].x*w
                y_little_base = dots[17][1].y*h

                # расстояние между 0 и 17
                dist_0_17 = int(math.sqrt(pow(x_little_base - x_wrist, 2) + pow(y_little_base - y_wrist, 2)))
                # растояние между большим и указательным
                delta_thumb_forefinger= int(math.sqrt(pow(x_forefinger - x_big, 2) + pow(y_forefinger - y_big, 2)))
                # расстояние между большим и средним
                delta_thumb_middle = int(math.sqrt(pow(x_middle - x_big, 2) + pow(y_middle - y_big, 2)))
                # расстояние между большим и безымянным
                delta_thumb_ring = int(math.sqrt(pow(x_ring - x_big, 2) + pow(y_ring - y_big, 2)))
                # расстояние между большим и мизинцем
                delta_thumb_little = int(math.sqrt(pow(x_little - x_big, 2) + pow(y_little - y_big, 2)))
                # расстояние между указательным и его основанием
                delta_forefinger_base = int(math.sqrt(pow(x_forefinger_base - x_forefinger, 2) + pow(y_forefinger_base - y_forefinger, 2)))
                # расстояние между большим и его основанием
                delta_thumb_base = int(math.sqrt(pow(x_big - x_big_base, 2) + pow(y_big -y_big_base, 2)))
                # расстояние между средним и его основанием
                delta_middle_base = int(math.sqrt(pow(x_middle - x_middle_base, 2) + pow(y_middle - y_middle_base,2)))
                # растояние между кончика мизинца и запясьтем
                d_little_wrist = int(math.sqrt(pow(x_wrist -x_little, 2) + pow(y_wrist - y_little, 2)))
                # расстояние между кочника мизинца и его основанием
                d_little_base = int(math.sqrt(pow(x_little - x_little_base, 2) + pow(y_little - y_little_base, 2)))
                # расстояние между кончиками указательного и среднего палец
                d_foref_middl = int(math.sqrt(pow(x_forefinger - x_middle, 2) + pow(y_forefinger - y_middle, 2)))


                # проценты от расстояния от кончика большого пальца и до его основания
                p_5_t_b = delta_thumb_base / 100 * 5
                p_6_t_b = delta_thumb_base / 100 * 6
                p_10_t_b = delta_thumb_base / 100 * 10

                # точка от 0 до 17 взять за основу для сравнения
                cv2.putText(image, f'dist_0_17 - {dist_0_17}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                            2, cv2.LINE_AA)

                # cv2.putText(image, f'{delta_thumb_forefinger}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),
                #             2, cv2.LINE_AA)

                # буква А
                if d_little_base <45 :
                    cv2.putText(image, f'A', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),
                                2, cv2.LINE_AA)

                # буква Б
                if d_foref_middl<55 and delta_thumb_ring<50 and delta_thumb_little<50 and delta_thumb_forefinger>120:
                    font = cv2.FONT_HERSHEY_COMPLEX
                    cv2.putText(image, 'Б', (100, 300), font, 3, color=(255, 255, 255), thickness=2)

                # буква О
                # fract = int(dist_0_17 / delta_thumb_forefinger)
                if delta_thumb_forefinger < 32 and delta_thumb_ring >45 and delta_thumb_middle > 45\
                        and delta_thumb_little > 50:
                    cv2.putText(image, 'O', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),
                            2, cv2.LINE_AA)
                cv2.putText(image, f'dist_th_frf-{delta_thumb_forefinger}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                            2, cv2.LINE_AA)
                # cv2.putText(image, f'fract-{fract}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                #             2, cv2.LINE_AA)

                # буква Р
                if delta_thumb_middle < 32 and delta_thumb_forefinger > 45 and delta_thumb_ring>45:
                    cv2.putText(image, 'P', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),
                            2, cv2.LINE_AA)
                # буква Н
                if delta_thumb_ring < 32 and delta_thumb_middle > 45 and delta_thumb_forefinger > 45 :
                    cv2.putText(image, 'H', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),
                            2, cv2.LINE_AA)

                # буква E
                if delta_thumb_ring < 34 and delta_thumb_middle < 34 and delta_thumb_forefinger < 34\
                        and delta_thumb_little < 34:
                    cv2.putText(image, 'E', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),
                            2, cv2.LINE_AA)

                # буква Г
                dot_3_x = dots[3][1].x * w
                dot_3_y = dots[3][1].y * h
                dot_4_x = dots[4][1].x * w
                dot_4_y = dots[4][1].y * h
                dot_5_x = dots[5][1].x * w
                dot_5_y = dots[5][1].y * h
                dot_6_x = dots[6][1].x * w
                dot_6_y = dots[6][1].y * h
                dot_12_x = dots[12][1].x * h
                dot_12_y = dots[12][1].y * h
                dot_16_x = dots[16][1].x * h
                dot_16_y = dots[16][1].y * h
                dot_20_x = dots[20][1].x * h
                dot_20_y = dots[20][1].y * h
                dot_10_x = dots[10][1].x * h
                dot_10_y = dots[10][1].y * h
                dot_14_x = dots[14][1].x * h
                dot_14_y = dots[14][1].y * h
                dot_18_x = dots[18][1].x * h
                dot_18_y = dots[18][1].y * h
                vect_43 = [dot_4_x - dot_3_x, dot_4_y - dot_3_y]
                vect_65 = [dot_6_x - dot_5_x, dot_6_y - dot_5_y]
                dis_vect_43 = math.sqrt(  pow(vect_43[0],2) + pow(vect_43[1],2)  )
                dis_vect_65 = math.sqrt(  pow(vect_65[0],2) + pow(vect_65[1],2)  )
                scal_vect_43_65 = (
                    vect_43[0] * vect_65[0]  + vect_43[1] * vect_65[1] ) / ( dis_vect_43 * dis_vect_65 )
                x = np.arccos([scal_vect_43_65])
                y = x * 180 / math.pi
                cv2.putText(image, f'angle-{y}', (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                            2, cv2.LINE_AA)
                if 55<y<105 and dot_12_y<dot_10_y and dot_16_y<dot_14_y and dot_20_y<dot_18_y\
                        and y_forefinger > dot_10_y: #
                    cv2.putText(image, 'G', (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2, cv2.LINE_AA)


                # мягкий знак
                if 55 < y < 105 and x_forefinger < dot_10_x and dot_12_x<dot_10_x and dot_16_x<dot_14_x\
                        and dot_20_x<dot_18_x:
                    cv2.putText(image, 'b', (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2, cv2.LINE_AA)
                    print('мяг згнак')

                # кончики другиз пальцев назрдятся между фалангами который выше
                # угол прмиерно 50-110 градусов
                #  кончик указательного ниже всех других точек других пальцев



            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

