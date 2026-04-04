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
 return np.exp(1j * k * z)


def b_n_vectorized(b, p):
 return b / (p * np.pi)


def K_plus_vectorized(alpha, k, b, N=100):
 n_half = np.arange(1, N +1) -0.5
 n_full = np.arange(1, N +1)
    
 b_n_h = b_n_vectorized(b, n_half)
 b_n_f = b_n_vectorized(b, n_full)
    
 sqrt_half = np.sqrt(1 - k**2 * b_n_h**2)
 sqrt_full = np.sqrt(1 - k**2 * b_n_f**2)
    
 factor_half = sqrt_half -1j * alpha * b_n_h
 factor_full = sqrt_full -1j * alpha * b_n_f
    
 res = np.prod(factor_half / factor_full)
 return res


def gamma_n_vectorized(n, k, b):
 n = np.asarray(n)
 return np.sqrt((n * np.pi / b)**2 - k**2)


def P_0_vectorized(k):
 Kp = K_plus_vectorized(k, k, b)
 return 0.5 * (1 + Kp**(-2))


def P_n_vectorized(n, k, b):
 n = np.asarray(n)
 gamma = gamma_n_vectorized(n, k, b)
 Kp = K_plus_vectorized(k, k, b)
 Kp_imag = K_plus_vectorized(1j * gamma, k, b)
 return (-1)**(n+1) * (1j * k) / (gamma * Kp * Kp_imag)


def P_n_half_vectorized(n, k, b):
 n = np.asarray(n)
 n_half = n -0.5
 gamma = gamma_n_vectorized(n_half, k, b)
 Kp = K_plus_vectorized(k, k, b)
 Kp_imag = K_plus_vectorized(1j * gamma, k, b)
 return (-1)**(n+1) * (1j * k * Kp_imag) / (n_half * np.pi * gamma * Kp)


def S_vectorized(y, z, k, b, N=100):
 n = np.arange(1, N +1)
 n_half = n -0.5
    
 P_n_vals = P_n_vectorized(n, k, b)
 P_n_half_vals = P_n_half_vectorized(n, k, b)
 gamma_full = gamma_n_vectorized(n, k, b)
 gamma_half = gamma_n_vectorized(n_half, k, b)
    
 z_abs = np.abs(z)
    
 exp_full = np.exp(-gamma_full[:, np.newaxis, np.newaxis] * z_abs[np.newaxis, :, :])
 exp_half = np.exp(-gamma_half[:, np.newaxis, np.newaxis] * z_abs[np.newaxis, :, :])
    
 cos_full = np.cos(n[:, np.newaxis, np.newaxis] * np.pi * y[np.newaxis, :, :] / b)
 cos_half = np.cos(n_half[:, np.newaxis, np.newaxis] * np.pi * y[np.newaxis, :, :] / b)
    
 term_full = P_n_vals[:, np.newaxis, np.newaxis] * cos_full * exp_full
 term_half = P_n_half_vals[:, np.newaxis, np.newaxis] * cos_half * exp_half
    
 res = np.sum(term_full + term_half, axis=0)
 return res


def phi_t_vectorized(y, z, k, b, N=100):
 z_positive = z >=0
    
 S_vals = S_vectorized(y, z, k, b, N)
    
 P0 = P_0_vectorized(k)
    
 exp_fwd = np.exp(1j * k * z)
 exp_bwd = np.exp(-1j * k * z)
    
 result_pos = P0 * exp_fwd + S_vals
 result_neg = exp_fwd + (1 - P0) * exp_bwd - S_vals
    
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
