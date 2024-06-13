import matplotlib.pyplot as plt

brands = ['Mazda', 'Toyota', 'Lexus', 'Buick', 'Honda', 'Hyundai', 'Ram', 'Subaru', 'Porsche', 'Dodge',
          'Infiniti', 'BMW', 'Nissan', 'Audi', 'Kia', 'GMC', 'Chevrolet', 'Volvo', 'Jeep', 'Mercedes-Benz',
          'Cadillac', 'Ford', 'Mini', 'Volkswagen', 'Tesla', 'Lincoln']
scores = [83, 74, 71, 70, 63, 62, 58, 57, 55, 54, 54, 52, 51, 46, 45, 43, 42, 41, 41, 40, 38, 38, 37, 36, 29, 8]

fig, ax = plt.subplots(figsize=(8, 12))
ax.barh(brands, scores, color='skyblue', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Reliability Score')
ax.set_ylabel('Car Brands')
ax.set_title("America's Top-Scoring Car Brands For Reliability", fontsize=14, fontweight='bold')
ax.grid(axis='x', linestyle='--', linewidth=0.5)
ax.invert_yaxis()  # Invert the y-axis to match the original image

# Add scores as labels at the end of each bar
for i, score in enumerate(scores):
    ax.text(score + 1, i, str(score), va='center', fontsize=10)

plt.tight_layout()
plt.show()