import tkinter as tk
from tkinter import ttk

def calculate_receivers(traffic, max_receivers):
    if traffic < 20:
        return 1
    elif traffic < 50:
        return min(2, max_receivers)
    elif traffic < 80:
        return min(3, max_receivers)
    else:
        return max_receivers



root = tk.Tk()
root.title("RF Receiver Energy Optimization")
root.geometry("500x350")
root.resizable(False, False)


title = ttk.Label(
    root,
    text="RF Receiver Energy Optimization Simulator",
    font=("Arial", 14, "bold")
)
title.pack(pady=15)


traffic_frame = ttk.Frame(root)
traffic_frame.pack(pady=10)

traffic_label = ttk.Label(traffic_frame, text="Traffic load")
traffic_label.grid(row=0, column=0, padx=5)

traffic_slider = ttk.Scale(
    traffic_frame,
    from_=0,
    to=100,
    orient="horizontal",
    length=300
)
traffic_slider.set(30)
traffic_slider.grid(row=0, column=1)

traffic_value_label = ttk.Label(traffic_frame, text="30 Mbps")
traffic_value_label.grid(row=0, column=2, padx=5)


rx_frame = ttk.Frame(root)
rx_frame.pack(pady=10)

rx_label = ttk.Label(rx_frame, text="Max receivers")
rx_label.grid(row=0, column=0, padx=5)

rx_slider = ttk.Scale(
    rx_frame,
    from_=1,
    to=4,
    orient="horizontal",
    length=300
)
rx_slider.set(4)
rx_slider.grid(row=0, column=1)

rx_value_label = ttk.Label(rx_frame, text="4")
rx_value_label.grid(row=0, column=2, padx=5)


result_label = ttk.Label(
    root,
    text="Optimal active receivers: 2",
    font=("Arial", 12, "bold")
)
result_label.pack(pady=30)


def update_values(*args):
    traffic = traffic_slider.get()
    max_rx = rx_slider.get()

    optimal_rx = calculate_receivers(traffic, max_rx)

    traffic_value_label.config(text=f"{traffic:.1f} Mbps")
    rx_value_label.config(text=str(int(max_rx)))
    result_label.config(text=f"Optimal active receivers: {optimal_rx}")


traffic_slider.config(command=update_values)
rx_slider.config(command=update_values)

update_values()

root.mainloop()
