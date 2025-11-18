import pygame
import random
import numpy as np
import sys
import math

# ==========================================
# Configuration & Hyperparameters
# ==========================================
WIDTH = 600
HEIGHT = 600
FPS = 60

# Genetic Algorithm Parameters
POPULATION_SIZE = 50
MUTATION_RATE = 0.1
ELITISM_COUNT = 5  # Number of top performers to keep unchanged
WEIGHT_SHIFT_STRENGTH = 0.5
NODE_ADD_PROB = 0.05
CONN_ADD_PROB = 0.08

# Game Physics
PIPE_VELOCITY = 5
PIPE_FREQUENCY = 1500  # Milliseconds
GRAVITY = 0.6
LIFT = -10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 50, 200)
SKY_BLUE = (135, 206, 235)
GRAY = (200, 200, 200)

# ==========================================
# Neural Network Class (Custom Implementation)
# ==========================================
class Genome:
    def __init__(self):
        # Inputs: Bird Y, Bird Vel, Pipe Dist, Top Pipe Y, Bottom Pipe Y
        self.input_nodes = 5 
        self.output_nodes = 1 # Output: Jump (>0.5) or not
        self.hidden_nodes = 0
        
        # Weights represented as a dictionary {(in_node, out_node): weight}
        # Nodes are indexed: 0-4 (Inputs), 5 (Output), 6+ (Hidden)
        self.genes = {}
        self.fitness = 0.0
        
        # Initialize simple direct connections
        for i in range(self.input_nodes):
            self.genes[(i, self.input_nodes)] = np.random.uniform(-1, 1)

    def feed_forward(self, inputs):
        """Simple forward pass through the possibly non-layered topology."""
        # Reset node values
        node_values = {i: 0.0 for i in range(self.input_nodes + self.output_nodes + self.hidden_nodes)}
        
        # Set input values
        for i, val in enumerate(inputs):
            node_values[i] = val
            
        # We need to process nodes in a somewhat topological order, 
        # or simply iterate multiple times for recurrence stability (simplified here).
        # For this demo, we sort genes by input node index to approximate feed-forward flow.
        # A true NEAT implementation requires complex dependency sorting.
        # We will use a simplified activation approach:
        
        # Collect inputs for all target nodes
        temp_values = {key: 0.0 for key in node_values}
        
        # Apply weights
        for (in_node, out_node), weight in self.genes.items():
            if in_node in node_values:
                temp_values[out_node] += node_values[in_node] * weight
        
        # Activation function (Tanh) for hidden and output
        # Since we don't have layers, we just compute based on immediate inputs.
        # Note: This is a simplified "Direct" pass. True NEAT handles hidden layers recursively.
        # To allow hidden nodes to work effectively in this simple script, we treat hidden nodes
        # as an intermediate computation step.
        
        # 1. Compute Hidden Nodes
        hidden_indices = [i for i in range(self.input_nodes + 1, self.input_nodes + 1 + self.hidden_nodes)]
        for h in hidden_indices:
            # Sum inputs for this hidden node
            val = 0
            for (i, o), w in self.genes.items():
                if o == h:
                    val += node_values[i] * w
            node_values[h] = np.tanh(val)
            
        # 2. Compute Output Node
        out_idx = self.input_nodes
        val = 0
        for (i, o), w in self.genes.items():
            if o == out_idx:
                val += node_values[i] * w
        
        return np.tanh(val) # Output is between -1 and 1

    def mutate(self):
        # 1. Weight Mutation
        for key in self.genes:
            if random.random() < MUTATION_RATE:
                self.genes[key] += np.random.uniform(-WEIGHT_SHIFT_STRENGTH, WEIGHT_SHIFT_STRENGTH)
                
        # 2. Node Mutation (Add a hidden node)
        if random.random() < NODE_ADD_PROB:
            if not self.genes: return
            # Pick a random connection to split
            conn_key = random.choice(list(self.genes.keys()))
            in_n, out_n = conn_key
            weight = self.genes.pop(conn_key)
            
            new_node_idx = self.input_nodes + 1 + self.hidden_nodes
            self.hidden_nodes += 1
            
            # Add two new connections
            self.genes[(in_n, new_node_idx)] = 1.0
            self.genes[(new_node_idx, out_n)] = weight

        # 3. Connection Mutation (Add new connection)
        if random.random() < CONN_ADD_PROB:
            # Pick random input (Input or Hidden)
            possible_inputs = list(range(self.input_nodes)) + \
                              list(range(self.input_nodes + 1, self.input_nodes + 1 + self.hidden_nodes))
            
            # Pick random output (Hidden or Output)
            possible_outputs = [self.input_nodes] + \
                               list(range(self.input_nodes + 1, self.input_nodes + 1 + self.hidden_nodes))
            
            i = random.choice(possible_inputs)
            o = random.choice(possible_outputs)
            
            if i != o and (i, o) not in self.genes:
                self.genes[(i, o)] = np.random.uniform(-1, 1)

    def crossover(self, other):
        """Basic crossover: take matching genes randomly, disjoint genes from fitter parent."""
        child = Genome()
        # Inherit topology size from self (assuming self is fitter or primary)
        child.hidden_nodes = self.hidden_nodes
        
        all_genes = set(self.genes.keys()) | set(other.genes.keys())
        
        for key in all_genes:
            if key in self.genes and key in other.genes:
                # Matching gene - pick random parent
                child.genes[key] = self.genes[key] if random.random() > 0.5 else other.genes[key]
            elif key in self.genes:
                # Disjoint gene from self
                child.genes[key] = self.genes[key]
            # We generally ignore disjoint genes from the less fit parent in this simple version
            
        return child

# ==========================================
# Game Entities
# ==========================================
class Bird:
    def __init__(self, brain=None):
        self.x = 100
        self.y = HEIGHT // 2
        self.vel = 0
        self.radius = 15
        self.brain = brain if brain else Genome()
        self.alive = True
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.score = 0

    def update(self):
        self.vel += GRAVITY
        self.y += self.vel
        self.score += 1 # Survival reward

        # Boundaries
        if self.y > HEIGHT or self.y < 0:
            self.alive = False

    def jump(self):
        self.vel = LIFT

    def think(self, pipes):
        # Find closest pipe
        closest_pipe = None
        dist = float('inf')
        for p in pipes:
            d = p.x + p.width - self.x
            if 0 < d < dist:
                dist = d
                closest_pipe = p
        
        if closest_pipe:
            # Normalize inputs to -1 to 1 range roughly
            inputs = [
                self.y / HEIGHT,
                self.vel / 20,
                closest_pipe.x / WIDTH,
                closest_pipe.top_height / HEIGHT,
                closest_pipe.bottom_y / HEIGHT
            ]
            
            output = self.brain.feed_forward(inputs)
            if output > 0.5:
                self.jump()

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), self.radius, 1)

class Pipe:
    def __init__(self):
        self.x = WIDTH
        self.width = 70
        self.gap = 160
        self.top_height = random.randint(50, HEIGHT - self.gap - 50)
        self.bottom_y = self.top_height + self.gap
        self.passed = False

    def update(self):
        self.x -= PIPE_VELOCITY

    def draw(self, screen):
        # Top Pipe
        pygame.draw.rect(screen, GREEN, (self.x, 0, self.width, self.top_height))
        pygame.draw.rect(screen, BLACK, (self.x, 0, self.width, self.top_height), 2)
        
        # Bottom Pipe
        pygame.draw.rect(screen, GREEN, (self.x, self.bottom_y, self.width, HEIGHT - self.bottom_y))
        pygame.draw.rect(screen, BLACK, (self.x, self.bottom_y, self.width, HEIGHT - self.bottom_y), 2)

    def collide(self, bird):
        if (bird.x + bird.radius > self.x and bird.x - bird.radius < self.x + self.width):
            if (bird.y - bird.radius < self.top_height or bird.y + bird.radius > self.bottom_y):
                return True
        return False

# ==========================================
# Main Engine
# ==========================================
def run_game():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NeuroEvolution Flappy Bird AI")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 20)

    generation = 1
    birds = [Bird() for _ in range(POPULATION_SIZE)]
    pipes = [Pipe()]
    last_pipe_time = pygame.time.get_ticks()
    
    high_score = 0
    speed_multiplier = 1

    running = True
    while running:
        # Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    speed_multiplier = 50 # Fast forward
                if event.key == pygame.K_DOWN:
                    speed_multiplier = 1  # Normal speed

        # Game Logic Loop (can run multiple times per frame for fast forward)
        for _ in range(speed_multiplier):
            # Pipe Management
            current_time = pygame.time.get_ticks()
            if current_time - last_pipe_time > PIPE_FREQUENCY / (speed_multiplier if speed_multiplier == 1 else 1): 
                # Note: simple timer logic for speedup isn't perfect but works for demo
                pipes.append(Pipe())
                last_pipe_time = current_time

            # Count alive birds
            alive_birds = [b for b in birds if b.alive]
            
            if len(alive_birds) == 0:
                # Next Generation
                generation += 1
                birds = next_generation(birds)
                pipes = [Pipe()]
                last_pipe_time = pygame.time.get_ticks()
                break

            # Update Pipes
            for p in pipes:
                p.update()
                # Remove off-screen pipes
                if p.x + p.width < 0:
                    pipes.remove(p)
            
            # Update Birds
            for b in alive_birds:
                b.think(pipes)
                b.update()
                
                # Collision Check
                for p in pipes:
                    if p.collide(b):
                        b.alive = False
                        # Penalize crashing
                        b.score -= 5 

                # Score counting (passing pipes)
                # Simple logic: if bird exists, score goes up by survival (done in update)
                
                if b.score > high_score:
                    high_score = b.score

        # Drawing
        screen.fill(SKY_BLUE)
        
        for p in pipes:
            p.draw(screen)
            
        for b in birds:
            if b.alive:
                b.draw(screen)

        # HUD
        alive_count = len([b for b in birds if b.alive])
        info_text = [
            f"Generation: {generation}",
            f"Alive: {alive_count}/{POPULATION_SIZE}",
            f"High Score: {high_score}",
            f"Speed: {speed_multiplier}x (Hold UP to speed up)",
        ]
        
        for i, line in enumerate(info_text):
            text_surf = font.render(line, True, BLACK)
            screen.blit(text_surf, (10, 10 + i * 25))

        # Visualize Best Network (Corner)
        if alive_birds:
            best_bird = max(alive_birds, key=lambda x: x.score)
            draw_network(screen, best_bird.brain, 450, 50, 140, 100)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

def next_generation(old_birds):
    """Evolutionary algorithm logic."""
    # Calculate fitness (normalize)
    total_score = sum(b.score for b in old_birds)
    if total_score == 0: total_score = 1
    
    # Sort by score
    old_birds.sort(key=lambda x: x.score, reverse=True)
    
    new_birds = []
    
    # Elitism: Keep best performers exactly as they are
    for i in range(ELITISM_COUNT):
        new_birds.append(Bird(brain=old_birds[i].brain))
        new_birds[-1].color = (255, 215, 0) # Gold color for elites

    # Generate rest of population
    while len(new_birds) < POPULATION_SIZE:
        # Selection (Roulette Wheel simplified or Tournament)
        parent1 = select_parent(old_birds, total_score)
        parent2 = select_parent(old_birds, total_score)
        
        # Crossover
        child_brain = parent1.brain.crossover(parent2.brain)
        
        # Mutation
        child_brain.mutate()
        
        new_birds.append(Bird(brain=child_brain))
        
    return new_birds

def select_parent(birds, total_score):
    # Tournament Selection (safer than roulette for small scores)
    tournament = random.sample(birds, 3)
    return max(tournament, key=lambda x: x.score)

def draw_network(screen, genome, x, y, w, h):
    """Draws a tiny visualization of the neural network."""
    pygame.draw.rect(screen, WHITE, (x, y, w, h))
    pygame.draw.rect(screen, BLACK, (x, y, w, h), 1)
    
    input_x = x + 10
    output_x = x + w - 10
    
    # Coordinates for nodes
    node_positions = {}
    
    # Inputs
    for i in range(genome.input_nodes):
        ny = y + 10 + i * (h // (genome.input_nodes))
        node_positions[i] = (input_x, ny)
        pygame.draw.circle(screen, BLUE, (input_x, ny), 4)
        
    # Output
    out_y = y + h // 2
    node_positions[genome.input_nodes] = (output_x, out_y)
    pygame.draw.circle(screen, RED, (output_x, out_y), 4)
    
    # Hidden
    if genome.hidden_nodes > 0:
        hidden_x = x + w // 2
        for i in range(genome.hidden_nodes):
            idx = genome.input_nodes + 1 + i
            ny = y + 20 + i * 15
            node_positions[idx] = (hidden_x, ny)
            pygame.draw.circle(screen, GREEN, (hidden_x, ny), 3)

    # Connections
    for (in_n, out_n), weight in genome.genes.items():
        if in_n in node_positions and out_n in node_positions:
            start = node_positions[in_n]
            end = node_positions[out_n]
            color = BLACK if weight > 0 else RED
            width = max(1, int(abs(weight) * 2))
            pygame.draw.line(screen, color, start, end, width)

if __name__ == "__main__":
    run_game()