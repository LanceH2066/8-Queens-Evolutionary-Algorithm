import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Constants defining the problem setup
BOARD_SIZE = 8          # Size of the chessboard (8x8)
POPULATION_SIZE = 100   # Number of candidate solutions in each generation
MUTATION_RATE = 0.2     # Probability of a mutation occurring in a child solution
GENERATIONS = 1000      # Maximum number of generations for evolution
RUNS = 100              # Number of independent runs to assess optimization behavior

def random_board():
    return [random.randint(0, BOARD_SIZE - 1) for _ in range(BOARD_SIZE)]

# FITNESS FUNCTION
def fitness(board):
    non_attacking = 28  # Start with max possible non-attacking pairs
    for i in range(BOARD_SIZE):
        for j in range(i + 1, BOARD_SIZE):
            # Reduce score if queens attack each other horizontally or diagonally
            if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                non_attacking -= 1
    return non_attacking

# TOURNAMENT SELECTION
def select(population):
    tournament = random.sample(population, 5)  # Select 5 random individuals
    return max(tournament, key=lambda x: x[1])  # Return individual with highest fitness

# SINGLE-POINT CROSSOVER FUNCTION
def crossover(parent1, parent2):
    point = random.randint(1, BOARD_SIZE - 2)  # Ensure at least one gene from each parent
    return parent1[:point] + parent2[point:]

# MUTATION FUNCTION
def mutate(board):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, BOARD_SIZE - 1)  # Select a random column to modify
        board[index] = random.randint(0, BOARD_SIZE - 1)  # Assign a new random row
    return board

# EVOLUTIONARY ALGORITHM FUNCTION
def genetic_algorithm():
    # Initialize a population with random boards and compute their fitness
    population = [(random_board(), 0) for _ in range(POPULATION_SIZE)]
    population = [(board, fitness(board)) for board, _ in population]

    best_fitness_per_gen = []  # Track best fitness score over generations
    
    for generation in range(GENERATIONS):
        # Sort population by fitness in descending order
        population = sorted(population, key=lambda x: -x[1])
        best_fitness_per_gen.append(population[0][1])  # Store best fitness

        if population[0][1] == 28:  # If an optimal solution is found, terminate early
            break  
        
        new_population = population[:10]  # Keep the top 10 best solutions
        
        # Generate new offspring to fill the population
        while len(new_population) < POPULATION_SIZE:
            parent1 = select(population)[0]
            parent2 = select(population)[0]
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append((child, fitness(child)))
        
        population = new_population  # Update the population

    return population[0][1], best_fitness_per_gen  # Return best fitness and its progression

# Store fitness results for statistical analysis
final_fitness_values = []  # Stores final fitness of each run
best_fitness_per_run = []  # Stores fitness progression over generations

# Execute the genetic algorithm multiple times to observe performance
for _ in range(RUNS):
    final_fitness, best_fitness = genetic_algorithm()
    final_fitness_values.append(final_fitness)
    best_fitness_per_run.append(best_fitness)

# Create a figure with multiple subplots for visualizing results
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# --- PLOT 1: Final Fitness Distribution (Histogram) ---
ax = axes[0, 0]
bins = [26.5, 27.5, 28.5]  # Align bin centers to key values (27 and 28)
ax.hist(final_fitness_values, bins=bins, color='skyblue', edgecolor='black', rwidth=0.8, align='mid')
ax.axvline(x=28, color='green', linestyle='dashed', linewidth=2, label="Global Optimum (28)")
ax.axvline(x=27, color='red', linestyle='dashed', linewidth=2, label="Local Optimum (27)")
ax.set_xticks([26, 27, 28, 29])  # Show only relevant fitness scores
ax.set_xlabel("Final Fitness Score")
ax.set_ylabel("Frequency of Occurrence")
ax.set_title("Final Fitness Distribution Across Runs")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

# --- PLOT 2: Fitness Progress Over Generations ---
ax = axes[0, 1]
for i in range(RUNS):
    ax.plot(best_fitness_per_run[i], color='blue', alpha=0.3)  # Show fitness trends
ax.set_xticks(np.arange(0, GENERATIONS + 1, 50))  # Adjust x-axis for readability
ax.set_xlabel("Generation")
ax.set_ylabel("Fitness Score")
ax.set_title("Fitness Progress Over Generations")
ax.grid(linestyle="--", alpha=0.7)

# --- PLOT 3: Scatter Plot of Final Fitness per Run ---
ax = axes[1, 0]
jittered_x = np.arange(RUNS) + np.random.uniform(-0.2, 0.2, RUNS)  # Add slight jitter to avoid overlap
ax.scatter(jittered_x, final_fitness_values, color='blue', alpha=0.7, label="Final Fitness per Run")
ax.axhline(y=28, color='green', linestyle='dashed', linewidth=2, label="Global Optimum (28)")
ax.set_xlabel("Run Index (1-100)")
ax.set_ylabel("Final Fitness Score")
ax.set_title("Final Fitness Scores Across Runs")
ax.legend()
ax.grid(linestyle="--", alpha=0.7)

# --- PLOT 4: Evolutionary Search Space Coverage (Heatmap) ---
ax = axes[1, 1]
heatmap_data = np.zeros((BOARD_SIZE, BOARD_SIZE))  # Create an empty heatmap grid

# Populate heatmap data with queen placements over multiple runs
for _ in range(RUNS):
    board = random_board()
    for col, row in enumerate(board):
        heatmap_data[row, col] += 1  # Increment cell count for each placement

# Generate heatmap visualization of queen placement frequencies
sns.heatmap(heatmap_data, cmap="coolwarm", annot=False, ax=ax, vmin=0, vmax=np.max(heatmap_data))
ax.set_xlabel("Column Index")
ax.set_ylabel("Row Index")
ax.set_title("Heatmap of Queen Placements Across Runs")

# Adjust layout and display all plots
plt.tight_layout()
plt.show()
