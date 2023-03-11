# -*- coding: utf-8 -*-
import math
import numpy as np
import linalg

# определение коллинеарности векторов с помощью формулы косинуса угла между векторами
#передаем спискок с координатами точек. dots = [ A[2,3] , B[4,7] ,C[3,4], D[6,3] ]
def collinear(dots):

    #вычисляем векторы AB и СD.
    vect_AB = [dots[1][0]-dots[0][0], dots[1][1]-dots[0][1]]
    print('vect_AB=',vect_AB)
    vect_CD = [dots[3][0]-dots[2][0],dots[3][1]-dots[2][1]]
    print('vect_CD=',vect_CD)

    # вычисляем длины векторов AB и СD.
    dist_AB = math.sqrt(pow(vect_AB[0], 2) + pow(vect_AB[1], 2))
    print('lenght_AB=',dist_AB)
    dist_CD = math.sqrt(pow(vect_CD[0], 2) + pow(vect_CD[1], 2))
    print('lenght_CD=',dist_CD)
    # угол между векторами AB и СD.
    angle_AB_CD = ( vect_AB[0] * vect_CD[0] + vect_AB[1] * vect_CD[1] ) / ( dist_AB * dist_CD )
    print(angle_AB_CD)
    x = np.arccos([angle_AB_CD])
    y = x * 180 / math.pi
    print('angle_AB_CD=',y)
    if angle_AB_CD == -1 or angle_AB_CD == 1:
        print('векторы коллинеарны')
    else:
        print('векторы линейно независимы')


a = [ [1,-1],[3,-1], [-1,-3], [-4,-3] ] # колллинеарны,противонаправлены
b = [ [2,2],[2,4],[-2,2],[-5,2]] # ортогональны
c = [ [1,-1],[3,-1],  [-4,-3],[-1,-3] ] # колллинеарны,сонаправлены
d = [ [2,-3],[4,-3],[1,2],[2,3] ] # 45 градусов
# collinear(dots=c)

# проверяем коллинеарность методом пропорции
# если пропорциональны ,то векторы коллинеарны
def collinear_1(dots):

    #вычисляем векторы AB и СD.
    vect_AB = [dots[1][0]-dots[0][0], dots[1][1]-dots[0][1]]
    print('vect_AB=',vect_AB)
    vect_CD = [dots[3][0]-dots[2][0],dots[3][1]-dots[2][1]]
    print('vect_CD=',vect_CD)
    # vect_AB = [3,7]
    # vect_CD = [-6,14]
    alpha_1 =  vect_AB[0]/vect_CD[0]
    alpha_2 = vect_AB[1]/vect_CD[1]
    if alpha_1 == alpha_2:
        print('векторы коллинеарны')
    else:
        print('векторы не коллинеарны')
# collinear_1(dots=a)

# методом определителя матрицы. Если определитель 0,то векторы коллинеарны
def collinear_2(dots):
    # вычисляем векторы AB и СD.
    vect_AB = [dots[1][0] - dots[0][0], dots[1][1] - dots[0][1]]
    print('vect_AB=', vect_AB)
    vect_CD = [dots[3][0] - dots[2][0], dots[3][1] - dots[2][1]]
    print('vect_CD=', vect_CD)

    det = vect_AB[0]*vect_CD[1]-vect_AB[1]*vect_CD[0]
    print(det)
collinear_2(dots=b)
