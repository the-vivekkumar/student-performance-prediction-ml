import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.linear_model import LinearRegression

# Data (0 to 10 hours → 0 to 100 marks)
hours = np.array([0,1,2,3,4,5,6,7,8,9,10]).reshape(-1, 1)
marks = np.array([0,10,20,30,40,50,60,70,80,90,100])

# Train model
model = LinearRegression()
model.fit(hours, marks)

# Figure
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Scatter + regression line
ax.scatter(hours, marks, s=40)
x_line = np.linspace(0, 10, 100).reshape(-1, 1)
ax.plot(x_line, model.predict(x_line), linewidth=2)

# Axis padding (THIS fixes the touching issue)
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-5, 105)

# Labels & title
ax.set_xlabel("Study Hours", fontsize=11)
ax.set_ylabel("Expected Marks", fontsize=11)
ax.set_title("Study Time and Performance Prediction", fontsize=13)

# Initial point
initial_hour = 3
initial_mark = model.predict([[initial_hour]])[0]

prediction_dot, = ax.plot(
    [initial_hour],
    [initial_mark],
    'o',
    color='green',
    markersize=10
)

# Slider
slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
hour_slider = Slider(slider_ax, "Hours", 0, 10, valinit=initial_hour)

# Tooltip annotation
annotation = ax.annotate(
    "",
    xy=(0, 0),
    xytext=(10, 10),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="lightyellow")
)
annotation.set_visible(False)

# Update dot
def update(val):
    h = hour_slider.val
    y = model.predict([[h]])[0]
    prediction_dot.set_data([h], [y])
    fig.canvas.draw_idle()

hour_slider.on_changed(update)

# Hover logic
def on_move(event):
    if event.inaxes == ax:
        x, y = prediction_dot.get_data()
        x, y = x[0], y[0]

        if abs(event.xdata - x) < 0.3 and abs(event.ydata - y) < 3:
            annotation.xy = (x, y)
            annotation.set_text(f"Hours: {x:.2f}\nMarks: {y:.2f}")
            annotation.set_visible(True)
        else:
            annotation.set_visible(False)

        fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_move)

plt.show()
