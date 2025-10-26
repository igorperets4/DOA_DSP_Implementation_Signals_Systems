"""
Signals and Systems
Author: Igor Perets | ID: 323454405

Stages in this file:
1) infrastructure – exponent matrix , FT/IFT,FS , Discrete convolution etc.
2) signals      – functions for each input
3) sampling     – ZOH helpers , delta train
4) plotting     – lightweight plotting helpers
5) Q1 setup     – essential functions -  (tau, W(α) vectorized etc.), uniqueness analysis
6) Q2 setup     - build essential functions.
6) runners      – run_question_1(), run_question_2()
7) main         – prepares axes, exponent, calls runners, one plt.show()
"""

from typing import Tuple, Callable
import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================================================
# (1) Fourier – exponents, FT/IFT
# ============================================================================
#check for commit
def prepare_exponents(t: np.ndarray, omega: np.ndarray):
    """
    Builds the exponential matrices E_minus (FT) and E_plus (IFT).

    :param:
        t: Time axis array.
        omega: Frequency axis array.

    :return:
        E_minus: Matrix for FT (e^{-jwt}), shape (w x t).
        E_plus: Matrix for IFT (e^{+jwt}), shape (t × w)
    """
    TW = np.outer(omega, t)         # (len w  × len t outer product)
    E_minus = np.exp(-1j * TW)      # for FT
    E_plus  = np.exp( 1j * TW.T)    # for IFT the transpose of TW
    return E_minus, E_plus

def continuous_fourier_transform(x_t: np.ndarray, dt: float, E_minus: np.ndarray):
    """
    Computes Continuous-Time Fourier Transform (FT) numerically
    Uses matrix multiplication for summation.

    :param:
        x_t: Time domain signal samples.
        dt: Time step
        E_minus: e^-jwt

    :return:
        Fourier Transform samples X(jw).
    """
    return (E_minus @ x_t) * dt # Matrix (w x t) * Vector (t x 1) = Vector (w x 1)

def inverse_fourier_transform(X_omega: np.ndarray, dw: float, E_plus: np.ndarray) :
    """
    Computes Inverse Fourier Transform (IFT) numerically.
    :param:
        X_omega: Frequency domain signal samples.
        dw: Frequency step (for integration scaling).
        E_plus: Precomputed positive exponential (e^{+jωt}).

    :returns:
        Time domain signal samples x(t) (real part only).
    """
    x_t = (E_plus @ X_omega) * (dw / (2.0 * math.pi)) # Matrix (t x w) X Vector (w x 1) = Vector (t x 1)
    return x_t.real

# ============================================================================
# (2) signals – basic signal generators (pure)
# ============================================================================

def create_time_axis(t_start: float, t_end: float, dt: float):
    """
    Creates a discrete, uniformly sampled time axis.

    :param:
        t_start: The starting point of the time axis [seconds].
        t_end: The ending point of the time axis [seconds].
        dt: The time step (sampling interval) [seconds].

    :return:
        An array representing the time samples (t).
    """
    num = int(round((t_end - t_start) / dt)) + 1
    return np.linspace(t_start, t_end, num) # num is the resolution

def create_frequency_axis(omega_min: float, omega_max: float, num_points: int):
    """
    Creates a discrete, uniformly sampled angular frequency axis.

    :param:
        omega_min: The minimum angular frequency [rad/s].
        omega_max: The maximum angular frequency [rad/s].
        num_points: The total number of discrete frequency samples.

    :returns:
        An array representing the angular frequency samples (omega).
    """
    return np.linspace(omega_min, omega_max, num_points) #num_points is the resolution


def rectangular_pulse(t: np.ndarray, width: float):
    """
    Generates a Rectangular Pulse, rect(t/width), centered at t=0.
    The pulse value is 1 for |t| <= width/2, and 0 otherwise.
    :param:
        t: The time axis array.
        width: The total duration (width) of the pulse.

    :return:
        An array containing the pulse signal (1s and 0s).
    """
    return (np.abs(t) <= width / 2.0).astype(float) # the astype converts True to 1 and False to 0


def cosine_signal(t: np.ndarray, amplitude: float, omega: float, phase: float = 0.0):
    """
    Generates a continuous-time cosine signal: A·cos(wt + phi).

    :param:
        t: The time axis array.
        amplitude: The maximum magnitude (A) of the signal.
        omega: The angular frequency (w) [rad/s].
        phase: The phase offset [radians] (default is 0).

    :return:
        An array containing the cosine waveform samples.
    """
    return amplitude * np.cos(omega * t + phase)


def dirac_delta_approximation(t: np.ndarray , dt) :
    """
    Generates a numerical approximation of the Dirac Delta function.

    The approximation is a unit impulse: 1 at the sample index closest to t=0,
    and 0 elsewhere.

    :param:
        t: The time axis array.
        dt: dt

    :return:
        An array representing the discrete impulse.
    """
    y = np.zeros_like(t, dtype=float)
    i0 = np.argmin(np.abs(t)) # closest to 0
    y[i0] = 1.0/dt
    return y

# ============================================================================
# (3) Q1 setup – Direction of Arrival math (DOA from now)
# ============================================================================

def tau_of_alpha(alpha_deg: float, d: float, c: float) :
    """
    calculate tao(a) via geometry

    :param  alpha_deg: the specific angle we check correlation
            d: distance between microphones
            c: speed of sound
    :return: the delay dependent on the angle [seconds]
    """
    return (d / c) * math.cos(math.radians(alpha_deg))


def time_shift_via_freq(X_omega: np.ndarray, omega: np.ndarray, tau: float):
    """
    Applies a time shift (delay) of tao to a signal in the frequency domain.
    This uses the Fourier Transform time-shifting property

    :param:
        X_omega: The original frequency spectrum X(jw).
        omega: The angular frequency axis (w).
        tau: The time delay (tao) to apply [s].

    Returns:
        The phase-shifted spectrum X_shift(jω).
    """
    return X_omega * np.exp(-1j * omega * tau)


def W_alpha_vec(Z1: np.ndarray, Z2: np.ndarray, omega: np.ndarray, d: float, c: float,
                     alpha_min: float = 0.0, alpha_max: float = 180.0, num_angles: int = 721
                     ) :
    """
    Computes W(a) over a range of angles.

    This function uses the Flancherel's theorem which says that the energy of a signal sustain between time and frequancy domain.

    :param:
        Z1: FT of sensor 1 signal.
        Z2: FT of sensor 2 signal.
        omega: Frequency axis.
        d: distance [m].
        c: Speed of sound [m/s].
        alpha_min, alpha_max: Angle range [deg].
        num_angles: Number of angles to evaluate.

    Returns:
        Tuple of (alphas: array of angles [deg], W_vals: array of correlation values W(α)).
    """
    alphas = np.linspace(alpha_min, alpha_max, num_angles) # alpha spectrum
    taus = (d / c) * np.cos(np.deg2rad(alphas)) # corresponding taus array
    base = Z2 * np.conjugate(Z1) # Z2 is already shifted by tao0
    E = np.exp(1j * np.outer(omega, taus)) # size (w x taus) matrix of all e^(j*w*tau)
    dw = omega[1] - omega[0]

    #shift by tao(a), positive exp due to conjugate
    # the @ satisfy the sum of integral due to matrix multiply
    W_vals = (base @ E) * (dw / (2.0 * math.pi)) # size (1 x tao) array of W(a)'s.
    return alphas, W_vals

def analyze_uniqueness(alphas: np.ndarray, W_vals: np.ndarray, alpha_true: float) -> dict:
    """
    Analyzes the W(α) function to determine if the true Angle of Arrival (α_true)
    can be uniquely estimated based on the magnitude of local maximum

    :param:
        alphas: Angle array [deg].
        W_vals: Correlation values W(α) (complex array).
        alpha_true: The known true angle [deg].

    :returns:
        A dictionary containing uniqueness status, number of significant peaks
        (ambiguities), and descriptive message.
    """
    mag = np.abs(W_vals)
    peaks = []

    # 1) find local peaks via checking each value with its neighbors
    for i in range(1, len(mag) - 1):
        if mag[i] > mag[i-1] and mag[i] > mag[i+1]:
            peaks.append({'angle': alphas[i], 'value': mag[i]})
    if not peaks:
        return {'unique': False, 'num_peaks': 0, 'message': 'No peaks found'} # if no local peaks.

    # 2) sort all the local peaks and find the greatest and set it to the global peak.
    p_sorted = sorted(peaks, key=lambda p: p['value'], reverse=True) # sort from big to small
    main = p_sorted[0]

    # 3) determine if there is an ambiguity in the correlation
    thr = 0.9 * main['value']
    sig = [p for p in peaks if p['value'] > thr]
    msg = f"Main peak at α={main['angle']:.1f}° (true {alpha_true:.1f}°), {len(sig)} significant peak(s)"
    return {'unique': len(sig) == 1, 'num_peaks': len(sig), 'main_peak_angle': main['angle'], 'message': msg} # return a dict with all the information

# ============================================================================
# (4) Q2 setup – FS coefficients, ZOH helpers , Q2 signals
# ============================================================================

def signal_x(t: np.ndarray, T1: float, A: float) -> np.ndarray:
    """
    Defines the base aperiodic signal x(t) = { 0 for 0 < t < T1, -A elsewhere }.

    :param:
        t: Time axis.
        T1: Duration of the zero-window.
        A: Magnitude of the signal elsewhere.

    :return:
        The aperiodic signal samples.
    """
    y = -A * np.ones_like(t)
    y[(t > 0) & (t < T1)] = 0.0
    return y


def signal_xT_periodic(t: np.ndarray, T: float, T1: float, A: float, num_periods: int = 10) :
    """
    Creates the periodic signal x_T(t) by summing shifted versions of x(t)
    over a finite number of periods.

    :param:
        t: Time axis.
        T: Period of the signal.
        T1, A: Parameters of the base signal x(t).
        num_periods: Number of periods to sum.

    :return:
        An array of the periodic signal samples x_T(t).
    """
    y = np.zeros_like(t, dtype=float)
    for k in range(-num_periods, num_periods + 1):
        y += signal_x(t - k * T, T1, A) + A
    return y - A


def H_jw(omega: np.ndarray) -> np.ndarray:
    """
    Defines the frequency response H(jw) for the Ideal HPF required in Q2.

    Analytically, H(jw) = 1/(1 - delta(w)), meaning H(jw) = 0 at w=0 and 1 elsewhere.
    This filter removes the DC component.

    :param:
        omega: Frequency axis.

    :return:
        The filter frequency response H(jw).
    """
    H = np.ones_like(omega)
    H[np.abs(omega) < 1e-3] = 0.0 # closest to zero
    return H



def compute_fourier_series(x_period_func: Callable[[np.ndarray], np.ndarray],
                           T: float, num_harmonics: int, t_samples: np.ndarray
                           ) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes the Fourier Series (FS) coefficients a_k numerically.

    :param:
        x_period_func: Function to take one period of the signal.
        T: Period of the signal.
        num_harmonics: Number of positive/negative harmonics to compute (N).
        t_samples: Time axis over one period.

    :return:
        Tuple of (a0: DC component, k_vals: harmonic indices k, ak: complex FS coefficients).
    """
    w0 = 2.0 * math.pi / T
    x_s = x_period_func(t_samples) # store one period of the function
    dt = t_samples[1] - t_samples[0]
    a0 = (1.0 / T) * np.sum(x_s) * dt # DC component e^j*0*t
    # Create two array: one for harmonics (k) one for coefficients (ak)
    k_vals = np.arange(-num_harmonics, num_harmonics + 1)
    ak = np.zeros_like(k_vals, dtype=complex)

    #  go on every k and map it to its coefficient
    for i, k in enumerate(k_vals):
        if k == 0:
            ak[i] = a0
        else:
            ak[i] = (1.0 / T) * np.sum(x_s * np.exp(-1j * k * w0 * t_samples)) * dt
    return a0, k_vals, ak


def create_zoh_impulse_response(Ts: float, dt: float) :
    """
    Creates the discrete ZOH impulse response h_ZOH[k] : a shifted window with width Ts.

    :params:
        Ts: Sampling period/pulse width [s].
        dt: Time step of the dense axis.

    :returns:
        discrete impulse response array.
    """
    M = int(round(Ts / dt))
    h_ZOH = np.ones(M, dtype=float)
    return h_ZOH


def create_sampled_impulse_train(y_samples: np.ndarray, Ts: float, dt: float) :
    """
    Creates the discrete impulse train, Train[k], by upsampling (factor L=Ts/dt).

    This represents the sampled signal, y[n], as a series of impulses
    on the time grid (dt), with zeros separating the samples.

    :params:
        y_samples: The sampled signal values y[n].
        Ts: Sampling period.
        dt: Time step of the fine output grid.

    Returns:
        The zero-padded impulse train array.
    """
    L = int(round(Ts / dt))
    len_train = (len(y_samples) - 1) * L + 1 # length same as time signal x(t)
    train = np.zeros(len_train, dtype=float)
    for i, val in enumerate(y_samples):
        train[i * L] = val # store evey value from sampled signal to its corresponding index
    return train

def discrete_convolution(x: np.ndarray, h: np.ndarray, dt: float):
    """
    Computes discrete convolution y[n] by stages:
    1. hold 1st signal and run second signal from -inf to inf
    2. multiply overlap values each iteration
    3. sum them and store the outcome in y[n] where n is the current iteration.

    :param:
        x: First signal (e.g., input or impulse train).
        h: Second signal (e.g., impulse response or ZOH filter).
        dt: Time step

    :returns:
        Tuple of (y: convolution result array, t_y: output time axis).
    """
    len_x, len_h = len(x), len(h)
    len_y = len_x + len_h - 1
    y = np.zeros(len_y, dtype=complex if (np.iscomplexobj(x) or np.iscomplexobj(h)) else float)

    # Core convolution sum
    for n in range(len_y):
        # set the index for summation for each iteration to ensure the limits of each sum
        k_min = max(0, n - (len_h - 1))
        k_max = min(n, len_x - 1)
        acc = 0.0 + 0.0j if np.iscomplexobj(y) else 0.0 # take both cases of complex and real
        for k in range(k_min, k_max + 1):
            acc += x[k] * h[n - k] # store the product in each acc value
        y[n] = acc

    t_y = np.arange(len_y) * dt # output time axis len = (lex(x) + len(h) -1)
    return y, t_y

# ============================================================================
# (5) plotting – helper functions
# ============================================================================

def plot_signal(t: np.ndarray, y: np.ndarray, title: str, color: str = 'b', linewidth: float = 2.0, ylabel: str = ''):
    """Plots a general time-domain signal.
    :param:
        t: The t axis
        y: corresponding y(t) values.
        title: Base title string.
    :return:
    """
    plt.figure(figsize=(10, 4))
    plt.plot(t, y, color=color, linewidth=linewidth)
    plt.title(title)
    plt.xlabel('Time [s]')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_spectrum(x_axis: np.ndarray, y_mag: np.ndarray, y_phase: np.ndarray, title_prefix: str, x_label: str,
                          plot_type: str = 'plot'):
    """
    Plots Magnitude and Phase spectra for FT (plot) or FS (stem).

    :param:
        x_axis: The x-axis data
        y_mag: Magnitude array.
        y_phase: Phase array.
        title_prefix: Base title string.
        x_label: X-axis label.
        plot_type: 'plot' for continuous FT, 'stem' for discrete FS.
    :return:
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    if plot_type == 'stem':
        plot_func_mag = lambda ax, x, y, c: ax.stem(x, y, basefmt=' ', linefmt=f'{c}-', markerfmt=f'{c}o')
        plot_func_phase = lambda ax, x, y, c: ax.stem(x, y, basefmt=' ', linefmt=f'{c}-', markerfmt=f'{c}o')
    else:
        plot_func_mag = lambda ax, x, y, c: ax.plot(x, y, f'{c}-', linewidth=1.5)
        plot_func_phase = lambda ax, x, y, c: ax.plot(x, y, f'{c}-', linewidth=1.5)

    # 1. Magnitude Plot
    plot_func_mag(axes[0], x_axis, y_mag, 'b')
    axes[0].set_title(f'{title_prefix} Magnitude')
    axes[0].set_xlabel(x_label);
    axes[0].set_ylabel('|·|');
    axes[0].grid(True, alpha=0.3)

    # 2. Phase Plot
    plot_func_phase(axes[1], x_axis, y_phase, 'r')
    axes[1].set_title(f'{title_prefix} Phase')
    axes[1].set_xlabel(x_label);
    axes[1].set_ylabel('∠ [rad]');
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_stem(x, y, title, xlabel='', ylabel=''):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stem(x, y, basefmt=' ')
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3); plt.tight_layout()
    return fig

# ============================================================================
# (6) runners – run_question_1(), run_question_2() (pure w.r.t. inputs)
# ============================================================================

def run_question_1(t: np.ndarray, dt: float, omega: np.ndarray, d: float, c: float, alpha_0: float,):
    """
    Runs Q1 given axes and physical params; returns nothing (plots + prints).

    :param:
        t: time axis
        dt: sample distance
        omega : omega axis
        d: distance between sensors
        c: speed of sound
        alpha_0: the angle of arrival
    :return
    """
    dw = omega[1] - omega[0]
    E_minus, E_plus = prepare_exponents(t, omega)
    test_cases_q1 = [
        (1, 'Dirac Delta', dirac_delta_approximation(t, dt), None),
        (2, 'Constant Signal (1)', np.ones_like(t), None),
        (3.1, f'Rectangular Pulse (Wide, Tr={0.1}s)', rectangular_pulse(t, width=0.1), 0.1), # 0.1 >> d/c = 1/340
        (3.2, f'Rectangular Pulse (Narrow, Tr={0.001}s)', rectangular_pulse(t, width=0.001), 0.001), #0.001<d/c=1/340
        (4, 'Cosine Signal w0 = 600', cosine_signal(t, amplitude=1.0, omega=600.0), None),
    ]

    for plot_idx, title, z1, Tr_val in test_cases_q1:

        Z1 = continuous_fourier_transform(z1, dt, E_minus)  # Z1(jw)
        tao0 = tau_of_alpha(alpha_0, d, c)
        Z2 = time_shift_via_freq(Z1, omega, tao0)  # Z2(jw) = Z1(jw) * e^-jw*tao0

        alphas, W_vals = W_alpha_vec(Z1, Z2, omega, d, c)  # store W(a) in array
        analysis = analyze_uniqueness(alphas, W_vals, alpha_0)

        # ----------------------------------------------------
        # Normalization for Dirac Delta only
        if plot_idx == 1:
            max_val = np.max(np.abs(W_vals))
            if max_val > 1e-10:
                W_vals = W_vals / max_val
        # ----------------------------------------------------

        plot_signal(t, z1.real, f"Q1-{plot_idx}: z₁(t) – {title}", ylabel='z₁(t)')  # plot z1(t)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(alphas, W_vals.real, label='W(α)')  # plot W(a)
        plt.axvline(alpha_0, linestyle='--', label=f'True α₀ = {alpha_0}°')
        imax = np.argmax(np.abs(W_vals))
        plt.plot(alphas[imax], np.abs(W_vals[imax]), 'o',
                 label=f'Estimated: {alphas[imax]:.1f}°')  # mark the maximum value
        plt.title(f"Q1-{plot_idx}: DOA Spectrum – {title}")
        plt.xlabel('Angle α [deg]')
        plt.ylabel('W(α)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()


        if title == 'Constant Signal (1)':
            plt.ylim(1.995, 2.005)  # restrict to mark the constant

        plt.show()
    # Printed uniqueness summary
    print("\n" + "═" * 70)
    print("QUESTION 1(c): UNIQUENESS ANALYSIS SUMMARY")
    print("═" * 70)
    print("\n1. DIRAC DELTA δ(t): ✓ Unique")
    print("2. CONSTANT SIGNAL: ✗ Not unique; W(α) ≈ constant")
    print("3. RECTANGULAR PULSE: ✓ Unique if 2T_r > d/c")
    print("4. COSINE: ⚠ Ambiguity when w0 > 1077 rad/sec \n")

def run_question_2(t: np.ndarray, dt: float, omega: np.ndarray, A: float, T1: float, T: float):
    """
    Runs Q2 given axes and params; returns nothing (plots + prints).
    """
    dw = omega[1] - omega[0]
    E_minus, E_plus = prepare_exponents(t, omega)

    # (a.1) x(t)
    x = signal_x(t, T1, A)
    X = continuous_fourier_transform(x, dt, E_minus)
    plot_signal(t, x, 'Q2(a.1): x(t)', ylabel='x(t)')
    X_mag = np.abs(X)
    X_phase = np.angle(X)
    plot_spectrum(omega, X_mag, X_phase, 'Q2(a.1): X(jω) –', 'ω [rad/s]')


    # (a.2) x_T(t) and FS
    xT = signal_xT_periodic(t, T, T1, A, num_periods=8)
    w0 = 2.0 * math.pi / T
    t_per = create_time_axis(0.0, T, dt) # create time in size T for FS
    x_one_period = lambda tao: signal_xT_periodic(tao, T, T1, A, num_periods=1) # create one period of a f
    a0, k_vals, ak = compute_fourier_series(x_one_period, T, num_harmonics=20, t_samples=t_per)
    print(f"   a0 = {a0:.6f},  w0 = 2π/T = {w0:.4f} rad/s")

    plot_signal(t, xT, 'Q2(a.2): x_T(t)', ylabel='x_T(t)')
    plot_stem(k_vals, np.abs(ak), 'Q2(a.2): |a_k|', xlabel='k', ylabel='|a_k|')
    plot_stem(k_vals, np.angle(ak), 'Q2(a.2): ∠a_k', xlabel='k', ylabel='Phase [rad]')

    # (a.3) h(t), H(jω)
    H = H_jw(omega)
    h = inverse_fourier_transform(H, dw, E_plus)  # ≈ δ(t) - 1 in numeric sense
    plot_signal(t, h, r'Q2(a.3): $h(t)$ (≈ $\delta(t)-1$)', ylabel='h(t)')
    plot_spectrum(omega, np.abs(H), np.angle(H), 'Q2(a.3): H(jω) –', 'ω [rad/s]')

    # (a.4) y(t) in freq domain
    Y = X * H
    y = inverse_fourier_transform(Y, dw, E_plus)
    plot_signal(t, y, r'Q2(a.4): $y(t)$ via IFT', ylabel='y(t)')
    plot_spectrum(omega, np.abs(Y), np.angle(Y), r'Q2(a.4): $Y(j\omega)$ –', 'ω [rad/s]')

    # (a.5) y_T(t) via FS: b_k = a_k·H(kw0) ⇒ remove DC
    ak_filt = ak.copy()  # work on a copy to keep 'ak' intact
    ak_filt[k_vals == 0] = 0.0  # H(0)=0 → kill DC ⇒ b0=0; for k≠0: H=1 ⇒ b_k=a_k
    b_k = ak_filt

    # read b0 safely (should be ~0 after filtering)
    b0 = (b_k[k_vals == 0][0])

    yT = xT - a0  # analytic result
    plot_signal(t, yT, r'Q2(a.5): $y_T(t)=x_T(t)-a_0$', ylabel=r'$y_T(t)$')
    plot_stem(k_vals * w0, np.abs(b_k) * 2 * math.pi,
              r'Q2(a.5): $|Y_T(j\omega)|$ (impulses, DC=0)',
              xlabel='ω [rad/s]', ylabel=r'$2\pi |b_k|$')
    plot_stem(k_vals * w0, np.angle(b_k), r'Q2(a.5): $\angle Y_T(j\omega)$',
              xlabel='ω [rad/s]', ylabel='Phase [rad]')



    # (b) Nyquist + ZOH reconstruction (explicit convolution on dense grid)

    # 1. Initialization and Setup
    t_dense = t
    yT_dense = yT
    dt_step = t_dense[1] - t_dense[0]

    # 2. ZOH Sampling Parameters
    Ts = T1 / 2.0  # Choose sampling period Ts = T1/2 to ensure perfect reconstruction (Ts <= T1)
    Fs = 1.0 / Ts  # Calculate sampling frequency Fs

    # sampling setup
    t0, t1 = t_dense[0], t_dense[-1]
    Ns = int(round((t1 - t0) / Ts)) + 1 # Number of samples
    t_samp = np.linspace(t0, t1, Ns)  # Create the discrete sampling time points

    # set y[n] as an array
    t_relative = t_samp - t_dense[0] # to start relative to 0
    indices = np.rint(t_relative / dt_step).astype(int)
    y_samp = yT_dense[indices] # y_samp is y[n]

    # 3. ZOH Convolution Preparation
    # The convolution is y_R[k] = Train[k] * h_zoh[k]
    h_zoh = create_zoh_impulse_response(Ts, dt_step)  # Create the causal ZOH pulse h[k]
    train = create_sampled_impulse_train(y_samp, Ts, dt_step)  # Create the upsampled impulse train
    y_rec_full, t_y_full = discrete_convolution(train, h_zoh, dt_step)  # Perform the explicit convolution

    # 4. Final Alignment and Plotting Data
    # Truncate the convolution result to the original dense axis length
    y_rec = y_rec_full[:len(t_dense)]

    fig = plt.figure(figsize=(10, 6))
    plt.plot(t_dense, yT_dense, label='Original $y_T(t)$', alpha=0.7)
    plt.plot(t_dense, y_rec, 'r--', label='ZOH Reconstruction')
    plt.plot(t_samp, y_samp, 'ko', markersize=4, label='Samples')
    plt.title(rf'Q2(b): ZOH Reconstruction via Convolution ($F_s={Fs:.1f}$ Hz)')
    plt.xlabel('Time [s]'); plt.ylabel(r'$y_T(t)$'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

# ============================================================================
# (7) main – sets params, calls runners (no globals changed)
# ============================================================================

def main():
    # Q1 parameters (keep as is - these are fine)
    c = 340.0
    d = 1
    alpha_0 = 90.0
    dt = 1e-3
    T_window = 2.0
    t = create_time_axis(-T_window / 2, T_window / 2, dt)
    omega = create_frequency_axis(-1000 * math.pi, 1000 * math.pi, 10001)

    run_question_1(t, dt, omega, d=d, c=c, alpha_0=alpha_0)

    # Q2 parameters - OPTIMIZED FOR SPEED:
    A = 5.0
    T1 = 3
    T = 5

    dt = T / 500  # 500 samples/period
    T_window_q2 = 3 * T  # 3 periods (was 20)
    t = create_time_axis(-T_window_q2 / 2, T_window_q2 / 2, dt)

    omega_max = 30 * (2 * np.pi / T)  # 30 harmonics
    num_omega = 3001
    omega = create_frequency_axis(-omega_max, omega_max, num_omega)

    run_question_2(t, dt, omega, A=A, T1=T1, T=T)

    plt.show()
if __name__ == "__main__":
    main()
