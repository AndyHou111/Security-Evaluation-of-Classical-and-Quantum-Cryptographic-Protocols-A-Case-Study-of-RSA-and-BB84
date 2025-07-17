import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

years = np.array([1998, 2000, 2006, 2008, 2016, 2018, 2019]).reshape(-1, 1)
qubits = np.array([2, 7, 12, 28, 50, 72, 128])

future_years = np.arange(2020, 2051).reshape(-1, 1)

def double_every_2years(years, start_year=2020, start_qubits=128):
    return start_qubits * 2 ** ((years - start_year) / 2)

double_2years_pred = double_every_2years(future_years.flatten(), start_year=2020, start_qubits=128)

poly3 = PolynomialFeatures(3)
years_poly3 = poly3.fit_transform(years)
model3 = LinearRegression()
model3.fit(years_poly3, qubits)
future_poly3 = poly3.transform(future_years)
pred3 = model3.predict(future_poly3)

def moore_law(years, start_year=2020, start_qubits=128, period=1.5):
    return start_qubits * 2 ** ((years - start_year) / period)

moore_pred = moore_law(future_years.flatten(), start_year=2020, start_qubits=128, period=1.5)

def double_every_year(years, start_year=2020, start_qubits=128):
    return start_qubits * 2 ** (years - start_year)

double_pred = double_every_year(future_years.flatten(), start_year=2020, start_qubits=128)

threshold = 20_000_000
def find_intersection(x, y, value):
    for i in range(1, len(y)):
        if (y[i-1] < value and y[i] >= value):
            x0, x1 = x[i-1], x[i]
            y0, y1 = y[i-1], y[i]
            year = x0 + (value - y0) * (x1 - x0) / (y1 - y0)
            return year
    return None

future_years_flat = future_years.flatten()
intersections = []
labels = ["Double Every 2 Years", "Polynomial Regression (Degree=3)", "Moore's Law (1.5 years double)", "Double Every Year"]
curves = [double_2years_pred, pred3, moore_pred, double_pred]
colors = ['blue', 'green', 'red', 'purple']

for curve in curves:
    year = find_intersection(future_years_flat, curve, threshold)
    intersections.append(year)

plt.figure(figsize=(13, 7))
plt.scatter(years, qubits, color='black', label='Historical Data', zorder=5)
plt.plot(future_years, double_2years_pred, color='blue', label='Double Every 2 Years', linewidth=2)
plt.plot(future_years, pred3, color='green', label='Polynomial Regression (Degree=3)', linewidth=2)
plt.plot(future_years, moore_pred, color='red', linestyle='--', label="Moore's Law (1.5 years double)", linewidth=2)
plt.plot(future_years, double_pred, color='purple', linestyle='--', label='Double Every Year', linewidth=2)

plt.axhline(threshold, color='orange', linestyle=':', linewidth=2, label='20,000,000 Qubits')

for idx, year in enumerate(intersections):
    if year is not None:
        yval = threshold
        plt.plot(year, yval, 'o', color=colors[idx], markersize=10)
        plt.axvline(year, color=colors[idx], linestyle=':', linewidth=1.5)
        plt.text(year+0.2, yval*0.8, f'{int(year)}', color=colors[idx], fontsize=12, rotation=90, va='bottom', ha='left')

plt.yscale('log')
plt.xlabel('Year')
plt.ylabel('Number of Qubits (log scale)')
plt.title('Qubit Growth Trend: Multiple Predictions (to 2050)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

for label, year in zip(labels, intersections):
    if year is not None:
        print(f"{label} crosses 20,000,000 qubits at year: {year:.2f}")
    else:
        print(f"{label} does not cross 20,000,000 qubits by 2050")
