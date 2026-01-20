import tkinter as tk
from tkinter import ttk
import random

NUM_STATIONS = 5
MAX_RECEIVERS = 4


def decide_state(traffic):
    if traffic < 20:
        return "SLEEP"
    elif traffic < 50:
        return "ECO"
    else:
        return "FULL"

def energy_consumption(state):
    if state == "SLEEP":
        return 0.2
    elif state == "ECO":
        return 0.6
    else:
        return 1.0


def update_station(index, value):
    traffic = float(value)
    state = decide_state(traffic)
    energy = energy_consumption(state)

    stations[index]['traffic_label'].config(text=f"{traffic:.1f} Mbps")
    stations[index]['state_label'].config(text=state)
    stations[index]['energy_label'].config(text=f"{energy:.2f}")

    update_total_energy()


def update_total_energy():
    total = sum(float(s['energy_label'].cget("text")) for s in stations)
    total_energy_label.config(text=f"Total Energy Consumption: {total:.2f}")

root = tk.Tk()
root.title("Multi-Base Station Energy Optimization Simulator")
root.geometry("750x450")
root.resizable(False, False)

title = ttk.Label(root, text="Multi-Base Station Energy Optimization",
                  font=("Arial", 14, "bold"))
title.pack(pady=10)


stations_frame = ttk.Frame(root)
stations_frame.pack(pady=10)

stations = []

for i in range(NUM_STATIONS):
    frame = ttk.LabelFrame(stations_frame, text=f"Base Station {i+1}")
    frame.grid(row=i//2, column=i%2, padx=10, pady=5, sticky="nsew")

    ttk.Label(frame, text="Traffic").grid(row=0, column=0)

    slider = ttk.Scale(
        frame, from_=0, to=100, orient="horizontal", length=200,
        command=lambda val, idx=i: update_station(idx, val)
    )
    slider.set(random.randint(10, 80))
    slider.grid(row=0, column=1, padx=5)

    traffic_label = ttk.Label(frame, text="0 Mbps")
    traffic_label.grid(row=0, column=2, padx=5)

    ttk.Label(frame, text="State:").grid(row=1, column=0)
    state_label = ttk.Label(frame, text="-")
    state_label.grid(row=1, column=1, sticky="w")

    ttk.Label(frame, text="Energy:").grid(row=2, column=0)
    energy_label = ttk.Label(frame, text="0.00")
    energy_label.grid(row=2, column=1, sticky="w")

    stations.append({
        'slider': slider,
        'traffic_label': traffic_label,
        'state_label': state_label,
        'energy_label': energy_label
    })

total_energy_label = ttk.Label(root,
    text="Total Energy Consumption: 0.00",
    font=("Arial", 12, "bold")
)
total_energy_label.pack(pady=20)

for i, station in enumerate(stations):
    update_station(i, station['slider'].get())

root.mainloop()
