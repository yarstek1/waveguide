"""
Оптимизированный расчёт рассеяния поршневой моды на поперечной стенке 
в бесконечном прямоугольном волноводе (векторные вычисления)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft

# Константы
c =5850
f =1000000
lmbd = c / f
k =2 * np.pi / lmbd +0j
b =0.04

# Сетка по z
z = np.linspace(-0.1,0.1,100, endpoint=False)
d_z = np.abs(z[0] - z[1])
alpha = fft.fftshift(fft.fftfreq(len(z), d_z))


def phi_i(z):
 """Векторизованная функция phi_i"""
 return np.exp(1j * k * z)


def b_n_vectorized(b, p):
 """Векторизованная функция b_n"""
 return b / (p * np.pi)


def K_plus_vectorized(alpha, k, b, N=100):
 """Векторизованная функция K_plus"""
 # Создаём массив n от1 до N
 n_half = np.arange(1, N +1) -0.5 # n -0.5
 n_full = np.arange(1, N +1) # n
    
 # Вычисляем b_n для всех n
 b_n_h = b_n_vectorized(b, n_half)
 b_n_f = b_n_vectorized(b, n_full)
    
 # Вычисляем корни один раз для всех n
 sqrt_half = np.sqrt(1 - k**2 * b_n_h**2)
 sqrt_full = np.sqrt(1 - k**2 * b_n_f**2)
    
 # Вычисляем множители
 factor_half = sqrt_half -1j * alpha * b_n_h
 factor_full = sqrt_full -1j * alpha * b_n_f
    
 # Произведение всех множителей
 res = np.prod(factor_half / factor_full)
 return res


def gamma_n_vectorized(n, k, b):
 """Векторизованная функция gamma_n"""
 n = np.asarray(n)
 return np.sqrt((n * np.pi / b)**2 - k**2)


def P_0_vectorized(k):
 """Векторизованная функция P_0"""
 Kp = K_plus_vectorized(k, k, b)
 return0.5 * (1 + Kp**(-2))


def P_n_vectorized(n, k, b):
 """Векторизованная функция P_n"""
 n = np.asarray(n)
 gamma = gamma_n_vectorized(n, k, b)
 Kp = K_plus_vectorized(k, k, b)
 Kp_imag = K_plus_vectorized(1j * gamma, k, b)
 return (-1)**(n+1) * (1j * k) / (gamma * Kp * Kp_imag)


def P_n_half_vectorized(n, k, b):
 """Векторизованная функция P_n_half"""
 n = np.asarray(n)
 n_half = n -0.5
 gamma = gamma_n_vectorized(n_half, k, b)
 Kp = K_plus_vectorized(k, k, b)
 Kp_imag = K_plus_vectorized(1j * gamma, k, b)
 return (-1)**(n+1) * (1j * k * Kp_imag) / (n_half * np.pi * gamma * Kp)


def S_vectorized(y, z, k, b, N=100):
 """
 Полностью векторизованная функция S(y, z)
 y, z -2D массивы (meshgrid)
 """
 # Массив n от1 до N
 n = np.arange(1, N +1)
 n_half = n -0.5
    
 # Вычисляем коэффициенты для всех n
 P_n_vals = P_n_vectorized(n, k, b) # (N,)
 P_n_half_vals = P_n_half_vectorized(n, k, b) # (N,)
 gamma_full = gamma_n_vectorized(n, k, b) # (N,)
 gamma_half = gamma_n_vectorized(n_half, k, b) # (N,)
    
 # |z| - абсолютное значение z (2D массив)
 z_abs = np.abs(z)
    
 # Вычисляем экспоненциальные множители
 # (N,1,1) * (1, Ny, Nz) -> (N, Ny, Nz)
 exp_full = np.exp(-gamma_full[:, np.newaxis, np.newaxis] * z_abs[np.newaxis, :, :])
 exp_half = np.exp(-gamma_half[:, np.newaxis, np.newaxis] * z_abs[np.newaxis, :, :])
    
 # Косинусные множители
 cos_full = np.cos(n[:, np.newaxis, np.newaxis] * np.pi * y[np.newaxis, :, :] / b)
 cos_half = np.cos(n_half[:, np.newaxis, np.newaxis] * np.pi * y[np.newaxis, :, :] / b)
    
 # Суммируем по n
 term_full = P_n_vals[:, np.newaxis, np.newaxis] * cos_full * exp_full
 term_half = P_n_half_vals[:, np.newaxis, np.newaxis] * cos_half * exp_half
    
 res = np.sum(term_full + term_half, axis=0)
 return res


def phi_t_vectorized(y, z, k, b, N=100):
 """
 Полностью векторизованная функция phi_t
 y, z -2D массива одинаковой формы
 """
 # Определяем знак z (True для z >=0)
 z_positive = z >=0
    
 # Вычисляем S(y, z) для всех точек
 S_vals = S_vectorized(y, z, k, b, N)
    
 # P_0 вычисляется один раз
 P0 = P_0_vectorized(k)
    
 # exp(1j * k * z) - для всех z
 exp_fwd = np.exp(1j * k * z)
 exp_bwd = np.exp(-1j * k * z) # для z< 0
    
 # Результат для z >=0
 result_pos = P0 * exp_fwd + S_vals
    
 # Результат для z< 0
 result_neg = exp_fwd + (1 - P0) * exp_bwd - S_vals
    
 # Комбинируем на основе знака z
 result = np.where(z_positive, result_pos, result_neg)
    
 return result


# === Построение тепловой карты ===
y = np.linspace(0,2 * b,100)
z = np.linspace(-0.1,0.1,100)
Z, Y = np.meshgrid(z, y)

Phi = phi_t_vectorized(Y, Z, k, b)
Phi_plot = np.abs(Phi)

plt.figure(figsize=(10,8))

plt.pcolormesh(Z, Y, Phi_plot, shading='auto', cmap='viridis')

plt.xlabel('z')
plt.ylabel('y')
plt.colorbar(label='abs(phi_t)')
plt.vlines(0,0, b, colors='red', linewidth=3.5)
plt.title('Тепловая карта функции phi_t (оптимизированная версия)')
plt.tight_layout()
plt.show()
