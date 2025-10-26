This project presents a comprehensive numerical implementation of two core areas in signal processing: **Direction of Arrival (DOA) estimation** and the **application of signal sampling and reconstruction processes**. The project demonstrates the ability to translate advanced theoretical knowledge (such as Fourier Transform, Plancherel's theorem, and Fourier Series) into functional, verifiable code solutions.

**Demonstrated Skills:**
* Mathematical and numerical modeling of continuous and discrete-time signal systems.
* Effective use of `NumPy` for fast, vectorized, and matrix computations.
* Analysis and visualization of simulation results using `Matplotlib`.

Modulde 1 : Direction of Arrival (DOA) Estimation

          Algorithm: Implementation of a DOA estimation technique based on Flancherel's theorem and the analysis of the cross-correlation function W(a).
          DSP Aspect: Utilized the Fourier Transform (FT) Time-Shift Property to apply time delay in the frequency domain via phase multiplication.
          Uniqueness Analysis: Examination of the conditions for unique angle-of-arrival detection (Ambiguity) based on signal type (rectangular pulse, cosine, Dirac Delta) and physical parameters

          For example: pure cosine input.  The given incoming anlge is 90 degrees.
<img width="778" height="682" alt="image" src="https://github.com/user-attachments/assets/73860609-be92-4a61-8724-af065b76c147" />
   As shown the output DOA is maximum at 90 degrees.


Module 2 : Frequency Analysis and ZOH Reconstruction

          Fourier Series (FS): Numerical computation of Fourier Series coefficients a_k for a periodic signal.
          Frequency Domain Filtering: Implementation of an Ideal High-Pass Filter (HPF) in the frequency domain H(jw) to remove the DC component of the signal.
          Sampling and Reconstruction: Modeling of the sampling process and Zero-Order Hold (ZOH) reconstruction using explicit, fundamental Discrete Convolution.

          ZOH Reconstruction: 

   <img width="1000" height="600" alt="Figure_3" src="https://github.com/user-attachments/assets/d8810d9f-e203-4fbd-b802-5be77cc9fedf" />
 As shown, the ZOH reconstruction maintain the value between each sample.





Author: Igor Perets
        3rd year EE student @ Bar - Ilan University.


