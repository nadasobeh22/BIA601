from flask import Flask, render_template, request
import random
import numpy as np
from deap import base, creator, tools, algorithms

app = Flask(__name__)

# Function to solve the Knapsack problem using Dynamic Programming
def knapsack(capacity, weights, values, n):
    K = [[0 for x in range(capacity + 1)] for x in range(n + 1)]
    for i in range(n + 1):
        for w in range(capacity + 1):
            if i == 0 or w == 0:
                K[i][w] = 0
            elif weights[i-1] <= w:
                K[i][w] = max(values[i-1] + K[i-1][w-weights[i-1]], K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
            # Debugging statement to check the matrix values
    return K[n][capacity]


# Function to create the distance matrix from user inputs
def create_distance_matrix(num_addresses, distances):
    size = num_addresses
    distance_matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(i + 1, size):
            distance_matrix[i][j] = distances[f'distance{i}_{j}']
            distance_matrix[j][i] = distance_matrix[i][j]
    return distance_matrix

# Genetic Algorithm setup for the TSP problem
def setup_ga_toolbox(addresses):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(addresses)), len(addresses))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalTSP(individual):
        distance = 0
        for i in range(len(individual)):
            distance += distance_matrix[individual[i-1]][individual[i]]
        return distance,

    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evalTSP)
    
    return toolbox

def tsp_main(toolbox):
    random.seed(42)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 100, stats=stats, halloffame=hof, verbose=True)

    best_route = hof[0]
    return best_route

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        # Get input data for knapsack problem
        capacity = int(request.form['capacity'])
        num_items = int(request.form['num_items'])
        weights = [int(request.form[f'weight{i}']) for i in range(num_items)]
        values = [int(request.form[f'value{i}']) for i in range(num_items)]

        # Validate inputs
        if any(w <= 0 for w in weights) or any(v <= 0 for v in values):
            return "Invalid input: weights and values must be positive numbers."

        # Debugging print statements to check input values
        print("Capacity:", capacity)
        print("Weights:", weights)
        print("Values:", values)

        # Solve knapsack problem
        max_value = knapsack(capacity, weights, values, num_items)
        print("Max value:", max_value)  # Debugging statement to check the result

        # Get input data for TSP problem
        num_addresses = int(request.form['num_addresses'])
        addresses = [request.form[f'address{i}'].strip() for i in range(num_addresses)]

        # Validate address inputs
        if len(set(addresses)) != num_addresses:
            return "Invalid input: addresses must be unique."

        # Get distances from user inputs
        distances = {f'distance{i}_{j}': int(request.form[f'distance{i}_{j}']) for i in range(num_addresses) for j in range(i + 1, num_addresses)}

        # Create distance matrix
        global distance_matrix
        distance_matrix = create_distance_matrix(num_addresses, distances)

        # Solve TSP problem
        toolbox = setup_ga_toolbox(addresses)
        best_route = tsp_main(toolbox)
        best_route_distance = toolbox.evaluate(best_route)[0]

        return render_template('result.html', max_value=max_value, best_route=best_route, best_route_distance=best_route_distance, addresses=addresses)
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)
