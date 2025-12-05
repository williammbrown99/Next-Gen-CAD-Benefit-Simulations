import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import beta
from matplotlib.colors import LightSource
from sklearn.linear_model import LinearRegression


# Define ranges
p = np.linspace(0, 0.6, 100)
alpha = np.linspace(0, 1, 100)
P, A = np.meshgrid(p, alpha)
QALDs_saved = 2445.5 * P * A

# Create figure
plt.figure(figsize=(7,6))

# Filled contour (smooth color gradient)
cf = plt.contourf(P, A, QALDs_saved, levels=50, cmap='plasma')
plt.colorbar(cf, label='QALDs saved per 100 patients')

# Overlay contour lines with labels
cs = plt.contour(P, A, QALDs_saved, levels=[200, 400, 600, 800, 1000], colors='white', linewidths=1.2)
plt.clabel(cs, fmt='%d', colors='white', fontsize=10)

# Highlight plausible range with a rectangle
#plt.gca().add_patch(plt.Rectangle((0.2,0.2), 0.2, 0.2, color='white', alpha=0.15, linestyle='--', edgecolor='black', linewidth=1.5))

# Axis labels and title
plt.xlabel('Overdiagnosis proportion $p$')
plt.ylabel('Next-gen CAD overdiagnosis identification $α$')
plt.title('Contour Plot of QALDs Saved per 100 Patients')
plt.tight_layout()
plt.show()



# --- Step 1: Define parameter distributions ---

# Overdiagnosis proportion p ~ Beta(a, b)
# Approximate mean 0.34, SD ~0.1
p_mean = 0.34
p_sd = 0.1
# Convert mean & SD to alpha, beta parameters
p_var = p_sd**2
p_alpha = ((1 - p_mean) / p_var - 1 / p_mean) * p_mean**2
p_beta = p_alpha * (1 / p_mean - 1)
# Safety: ensure parameters are >0
p_alpha = max(p_alpha, 0.01)
p_beta = max(p_beta, 0.01)

# CAD accuracy alpha ~ Beta(a, b)
# Approximate mean 0.3, SD ~0.05
a_mean = 0.3
a_sd = 0.05
a_var = a_sd**2
a_alpha = ((1 - a_mean) / a_var - 1 / a_mean) * a_mean**2
a_beta = a_alpha * (1 / a_mean - 1)
a_alpha = max(a_alpha, 0.01)
a_beta = max(a_beta, 0.01)

# --- Step 2: Monte Carlo draws ---
N = 10000
p_samples = beta.rvs(p_alpha, p_beta, size=N)
alpha_samples = beta.rvs(a_alpha, a_beta, size=N)

# --- Step 3: Calculate QALDs saved for each draw ---
QALDs_saved_samples = 2445.5 * p_samples * alpha_samples

# --- Step 4: Histogram of PSA outcomes ---
plt.figure(figsize=(7,5))
plt.hist(QALDs_saved_samples, bins=50, color='skyblue', alpha=0.7, density=True, edgecolor='black')

# Add vertical lines for mean, median, 95% CI
mean_qald = np.mean(QALDs_saved_samples)
median_qald = np.median(QALDs_saved_samples)
ci_lower = np.percentile(QALDs_saved_samples, 2.5)
ci_upper = np.percentile(QALDs_saved_samples, 97.5)

plt.axvline(mean_qald, color='red', linestyle='--', label=f'Mean = {mean_qald:.1f}')
plt.axvline(median_qald, color='green', linestyle='-.', label=f'Median = {median_qald:.1f}')
plt.axvline(ci_lower, color='purple', linestyle=':', label=f'95% CI = [{ci_lower:.1f}, {ci_upper:.1f}]')
plt.axvline(ci_upper, color='purple', linestyle=':')

plt.xlabel('QALDs saved per 100 patients')
plt.ylabel('Probability density')
plt.title('Probabilistic Sensitivity Analysis: QALDs Saved')
plt.legend()
plt.tight_layout()
plt.show()





# -----------------------------------------
# Core model parameters
# -----------------------------------------
od = np.arange(0.05, 0.61, 0.01)   # true biological overdiagnosis rate (5–60%)
L = 24.455                         # QALDs lost per overdiagnosed patient

# QALDs lost per 100 screened patients
q_none = np.zeros_like(od)
q_trad = od * L * 100
q_30   = od * (1 - 0.30) * L * 100
q_40   = od * (1 - 0.40) * L * 100
q_50   = od * (1 - 0.50) * L * 100


# ---------------------------------------------------
# FIGURE A — Deterministic (Lines Only)
# ---------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(od*100, q_none,  label="Screen None", linestyle="--", linewidth=2, color="black")
plt.plot(od*100, q_trad, label="Traditional CAD (0% OD detection)", linewidth=2, color="red")
plt.plot(od*100, q_30,   label="Next-Gen CAD (30% OD detection)", linewidth=3, color="green")
plt.plot(od*100, q_40,   label="Next-Gen CAD (40% OD detection)", linewidth=3, color="#87CEFA")
plt.plot(od*100, q_50,   label="Next-Gen CAD (50% OD detection)", linewidth=3, color="orange")

plt.xlabel("Overdiagnosis Rate (%)")
plt.ylabel("QALDs lost per 100 screened patients")
plt.title("Clinical Burden of Overdiagnosis Across Screening Strategies")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()