from flask import Flask, jsonify, render_template, request
import threading, time, random
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Konfigurasi simulasi
GRID_WIDTH = 50
GRID_HEIGHT = 50
GENOTYPE_LENGTH = 8
MUTATION_RATE = 0.01       # Dapat diubah secara real time
SIMULATION_DELAY = 0.1     # Delay (detik), bisa sampai 3 detik
TARGET_GENOTYPE = [1] * GENOTYPE_LENGTH

# Global variabel untuk simulasi
grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
generation = 0
stats = {"generation": generation, "average_fitness": 0, "best_fitness": 0}
history = []             # Menyimpan data historis (generation, average_fitness)
paused = False           # Status simulasi (running/paused)
species_bias = 0.5       # Probabilitas munculnya species 0 (default 50:50)

# Fungsi inisialisasi genotype dengan species berdasarkan species_bias
def random_genotype():
    return {"genotype": [random.randint(0, 1) for _ in range(GENOTYPE_LENGTH)],
            "species": 0 if random.random() < species_bias else 1}

# Fungsi fitness: jumlah bit yang sesuai dengan target, dikalibrasi oleh faktor species
def fitness(cell):
    genotype = cell["genotype"]
    base_fit = sum(1 for bit, target in zip(genotype, TARGET_GENOTYPE) if bit == target)
    # species 0 faktor 1.0, species 1 faktor 0.9
    factor = 1.0 if cell["species"] == 0 else 0.9
    return base_fit * factor

def select_parent(neighbors):
    total_fitness = sum(fitness(n) for n in neighbors) + 1e-6
    pick = random.uniform(0, total_fitness)
    current = 0
    for n in neighbors:
        current += fitness(n)
        if current >= pick:
            return n
    return random.choice(neighbors)

def crossover(parent1, parent2):
    p1 = parent1["genotype"]
    p2 = parent2["genotype"]
    point = random.randint(1, GENOTYPE_LENGTH - 1)
    child_genotype = p1[:point] + p2[point:]
    # Pewarisan species: pilih salah satu dari induk secara acak
    child_species = random.choice([parent1["species"], parent2["species"]])
    return {"genotype": child_genotype, "species": child_species}

def mutate(cell):
    global MUTATION_RATE
    new_genotype = [bit if random.random() > MUTATION_RATE else 1 - bit for bit in cell["genotype"]]
    return {"genotype": new_genotype, "species": cell["species"]}

def get_neighbors(x, y):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx = (x + dx) % GRID_WIDTH
            ny = (y + dy) % GRID_HEIGHT
            neighbors.append(grid[ny][nx])
    return neighbors

def simulation_loop():
    global grid, generation, stats, SIMULATION_DELAY, history, paused
    while True:
        if paused:
            time.sleep(SIMULATION_DELAY)
            continue

        new_grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        total_fit = 0
        best_fit = 0
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                neighbors = get_neighbors(x, y)
                parent1 = select_parent(neighbors)
                parent2 = select_parent(neighbors)
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_grid[y][x] = child
                f = fitness(child)
                total_fit += f
                if f > best_fit:
                    best_fit = f
        grid = new_grid
        generation += 1
        avg_fit = total_fit / (GRID_WIDTH * GRID_HEIGHT)
        stats = {"generation": generation, "average_fitness": avg_fit, "best_fitness": best_fit}
        history.append({"generation": generation, "average_fitness": avg_fit})
        if len(history) > 100:
            history.pop(0)
        time.sleep(SIMULATION_DELAY)

# Inisialisasi grid
for y in range(GRID_HEIGHT):
    for x in range(GRID_WIDTH):
        grid[y][x] = random_genotype()

# Jalankan simulasi pada thread background
sim_thread = threading.Thread(target=simulation_loop)
sim_thread.daemon = True
sim_thread.start()

# Endpoint untuk halaman utama
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/grid")
def api_grid():
    return jsonify(grid)

@app.route("/api/stats")
def api_stats():
    return jsonify(stats)

@app.route("/api/histogram")
def api_histogram():
    distribution = [0] * (GENOTYPE_LENGTH + 1)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            f = int(round(fitness(grid[y][x])))
            f = max(0, min(f, GENOTYPE_LENGTH))
            distribution[f] += 1
    return jsonify({"histogram": distribution})

@app.route("/api/prediction")
def api_prediction():
    if len(history) < 5:
        return jsonify({"prediction": None, "message": "Not enough data"})
    X = np.array([d["generation"] for d in history]).reshape(-1, 1)
    y_vals = np.array([d["average_fitness"] for d in history])
    model = LinearRegression()
    model.fit(X, y_vals)
    next_gen = generation + 1
    prediction = model.predict(np.array([[next_gen]]))[0]
    return jsonify({"prediction": prediction})

@app.route("/api/history")
def api_history():
    return jsonify(history)

# Endpoint untuk update parameter mutation_rate dan simulation_delay
@app.route("/api/set_parameters", methods=["POST"])
def set_parameters():
    global MUTATION_RATE, SIMULATION_DELAY
    data = request.get_json()
    if "mutation_rate" in data:
        try:
            MUTATION_RATE = float(data["mutation_rate"])
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid mutation rate value"}), 400
    if "simulation_delay" in data:
        try:
            SIMULATION_DELAY = float(data["simulation_delay"])
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid simulation delay value"}), 400
    print("Parameters updated: Mutation Rate =", MUTATION_RATE, "Simulation Delay =", SIMULATION_DELAY)
    return jsonify({"status": "success", "mutation_rate": MUTATION_RATE, "simulation_delay": SIMULATION_DELAY})

# Endpoint untuk mengatur preset mode
@app.route("/api/set_preset", methods=["POST"])
def set_preset():
    global MUTATION_RATE, SIMULATION_DELAY, species_bias
    data = request.get_json()
    preset = data.get("preset", "default")
    if preset == "default":
        MUTATION_RATE = 0.01
        SIMULATION_DELAY = 0.1
        species_bias = 0.5
    elif preset == "Mutasi Cepat":
        MUTATION_RATE = 0.05
        SIMULATION_DELAY = 0.2
        species_bias = 0.5
    elif preset == "Spesies Dominan":
        MUTATION_RATE = 0.01
        SIMULATION_DELAY = 0.1
        species_bias = 0.8
    elif preset == "Seleksi Ketat":
        MUTATION_RATE = 0.005
        SIMULATION_DELAY = 0.05
        species_bias = 0.5
    else:
        return jsonify({"status": "error", "message": "Unknown preset"}), 400
    print("Preset applied:", preset)
    return jsonify({"status": "success", "preset": preset, "mutation_rate": MUTATION_RATE, "simulation_delay": SIMULATION_DELAY, "species_bias": species_bias})

# Endpoint untuk toggle pause/play
@app.route("/api/toggle_pause", methods=["POST"])
def toggle_pause():
    global paused
    paused = not paused
    print("Paused status:", paused)
    return jsonify({"status": "success", "paused": paused})

# Endpoint untuk reset simulasi
@app.route("/api/reset", methods=["POST"])
def reset():
    global grid, generation, stats, history
    generation = 0
    stats = {"generation": generation, "average_fitness": 0, "best_fitness": 0}
    history = []
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            grid[y][x] = random_genotype()
    print("Simulation reset")
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)
