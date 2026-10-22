"""
Racing Game with Genetic Evolution
===================================
A simple genetic algorithm that trains neural networks to drive a car around a track.

How it works:
1. Each car has a "brain" (neural network) that decides how to drive
2. The brain receives sensor readings and outputs a driving decision
3. Cars that drive further are "more fit" and have more children
4. Children are copies of parents with small random changes (mutations)
5. Over many generations, cars learn to drive better

Author: Your name here
"""

import math
import random
from pathlib import Path

import pygame
from rays import ray_mask_collision

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def scale_image(image, scale_factor):
    """Make an image larger or smaller."""
    width, height = image.get_size()
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    return pygame.transform.scale(image, (new_width, new_height))


def rotate_and_draw(window, image, position, angle):
    """Rotate an image and draw it centered at a position."""
    rotated_image = pygame.transform.rotate(image, angle)
    # Get the rectangle for positioning
    rect = rotated_image.get_rect()
    # Center it on the car's position
    rect.center = image.get_rect(topleft=position).center
    window.blit(rotated_image, rect)


# ============================================================================
# GAME SETTINGS
# ============================================================================

# Window and track settings
pygame.init()
pygame.display.set_caption("Racing Game - Genetic Evolution")

TRACK_IMAGE = "imgs/track.png"
CAR_IMAGE = "imgs/car.png"
FINISH_IMAGE = "imgs/finish.png"
TRACK_SCALE = 0.9
CAR_SCALE = 0.4

# Load images
track = scale_image(pygame.image.load(TRACK_IMAGE), TRACK_SCALE)
track_mask = pygame.mask.from_surface(track)  # Used for collision detection
screen_width, screen_height = track.get_size()
screen = pygame.display.set_mode((screen_width, screen_height))

finish = pygame.image.load(FINISH_IMAGE)
car_image = scale_image(pygame.image.load(CAR_IMAGE), CAR_SCALE)

# Sensor settings - these "eyes" help the car see the track edges
SENSOR_ANGLES = tuple(range(-180, 181, 18))  # Angles from -180 to 180 degrees
SENSOR_MAX_DISTANCE = 250  # How far the sensors can "see"
SENSOR_STEP = 2  # How precisely to check for walls
SENSOR_COLOR = (0, 220, 80)  # Green color for sensor lines

# ============================================================================
# EVOLUTION SETTINGS
# ============================================================================

# Population settings
POPULATION_SIZE = 40  # Number of cars in each generation
PARENT_COUNT = 8  # Top performers that become parents

# Neural network settings
INPUT_NEURONS = len(SENSOR_ANGLES) + 1  # Sensors + current speed
HIDDEN_NEURONS = 8  # "Brain complexity" - more neurons = smarter but slower
OUTPUT_NEURONS = 4  # Number of possible actions

# Mutation settings (how much children change from parents)
BASE_MUTATION_RATE = 0.2  # Chance each weight changes (20%)
BASE_MUTATION_STRENGTH = 0.5  # How much weights change

# Stagnation settings (what happens when learning stops)
STAGNATION_GENERATIONS = 8  # If no improvement for this many generations

# Episode settings
MAX_FRAMES_PER_GENERATION = 1800  # Time limit per generation

# Fitness settings (how we score driving performance)
DISTANCE_REWARD = 0.2  # Points per pixel traveled
TIME_PENALTY = 0.1  # Points lost per frame alive (encourages speed)

# Circle detection (prevents driving in loops)
POSITION_HISTORY_SIZE = 100  # How many past positions to remember
MIN_FRAMES_FOR_CIRCLE = 30  # Minimum frames before checking for circles
CIRCLE_DISTANCE_THRESHOLD = 40.0  # Distance to consider "returned to same spot"
CIRCLE_PENALTY = 100.0  # Penalty for circling


# ============================================================================
# NEURAL NETWORK CLASS
# ============================================================================

class NeuralNetwork:
    """
    A simple "brain" for the car.
    
    The network has three layers:
    - Input layer: receives sensor data and speed
    - Hidden layer: processes the information
    - Output layer: decides what action to take
    
    Think of it like this:
    - Inputs are what the car "sees"
    - Hidden neurons are "thoughts"
    - Outputs are "decisions"
    """
    
    def __init__(self, num_inputs, num_hidden, num_outputs):
        """Create a new random neural network."""
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        # Create random weights (connections between neurons)
        # weights_input_to_hidden[i][h] = connection from input i to hidden h
        self.weights_input_to_hidden = [
            [random.gauss(0, 1) for _ in range(num_hidden)]
            for _ in range(num_inputs)
        ]
        
        # weights_hidden_to_output[h][o] = connection from hidden h to output o
        self.weights_hidden_to_output = [
            [random.gauss(0, 1) for _ in range(num_outputs)]
            for _ in range(num_hidden)
        ]
        
        # Bias values (like a "default" activation for each neuron)
        self.hidden_bias = [random.gauss(0, 1) for _ in range(num_hidden)]
        self.output_bias = [random.gauss(0, 1) for _ in range(num_outputs)]
    
    def think(self, inputs):
        """
        Make a decision based on inputs.
        
        Steps:
        1. Multiply inputs by weights to get hidden layer values
        2. Apply activation function (tanh) to hidden values
        3. Multiply hidden values by weights to get outputs
        4. Apply activation function to outputs
        5. Return the outputs
        """
        # Step 1 & 2: Calculate hidden layer
        hidden_values = []
        for h in range(self.num_hidden):
            # Start with bias
            total = self.hidden_bias[h]
            # Add weighted inputs
            for i in range(self.num_inputs):
                total += inputs[i] * self.weights_input_to_hidden[i][h]
            # Apply activation function (tanh squashes to -1 to 1)
            hidden_values.append(math.tanh(total))
        
        # Step 3 & 4: Calculate output layer
        output_values = []
        for o in range(self.num_outputs):
            # Start with bias
            total = self.output_bias[o]
            # Add weighted hidden values
            for h in range(self.num_hidden):
                total += hidden_values[h] * self.weights_hidden_to_output[h][o]
            # Apply activation function
            output_values.append(math.tanh(total))
        
        return output_values
    
    def copy(self):
        """Create an exact copy of this network."""
        new_brain = NeuralNetwork(self.num_inputs, self.num_hidden, self.num_outputs)
        
        # Copy all weights and biases
        new_brain.weights_input_to_hidden = [
            [weight for weight in row]
            for row in self.weights_input_to_hidden
        ]
        new_brain.weights_hidden_to_output = [
            [weight for weight in row]
            for row in self.weights_hidden_to_output
        ]
        new_brain.hidden_bias = [bias for bias in self.hidden_bias]
        new_brain.output_bias = [bias for bias in self.output_bias]
        
        return new_brain
    
    def mutate(self, mutation_rate, mutation_strength):
        """
        Randomly change some weights.
        
        mutation_rate: chance each weight changes (0.0 to 1.0)
        mutation_strength: how much weights can change
        """
        # Mutate input-to-hidden weights
        for i in range(self.num_inputs):
            for h in range(self.num_hidden):
                if random.random() < mutation_rate:
                    # Add a random change
                    change = random.gauss(0, mutation_strength)
                    self.weights_input_to_hidden[i][h] += change
        
        # Mutate hidden-to-output weights
        for h in range(self.num_hidden):
            for o in range(self.num_outputs):
                if random.random() < mutation_rate:
                    change = random.gauss(0, mutation_strength)
                    self.weights_hidden_to_output[h][o] += change
        
        # Mutate biases
        for h in range(self.num_hidden):
            if random.random() < mutation_rate:
                self.hidden_bias[h] += random.gauss(0, mutation_strength)
        
        for o in range(self.num_outputs):
            if random.random() < mutation_rate:
                self.output_bias[o] += random.gauss(0, mutation_strength)
    
    def get_weight_stats(self):
        """
        Get statistics about the weights.
        Returns the maximum weight value and the average weight value.
        """
        all_weights = []
        
        for row in self.weights_input_to_hidden:
            all_weights.extend(row)
        for row in self.weights_hidden_to_output:
            all_weights.extend(row)
        all_weights.extend(self.hidden_bias)
        all_weights.extend(self.output_bias)
        
        max_weight = max(abs(w) for w in all_weights)
        avg_weight = sum(abs(w) for w in all_weights) / len(all_weights)
        
        return max_weight, avg_weight
    
    def normalize_weights(self, max_allowed=5.0):
        """
        Normalize weights if any single weight becomes too strong.
        
        This prevents one neuron from dominating the decision making.
        If max weight exceeds max_allowed, scale all weights down proportionally.
        """
        max_weight, avg_weight = self.get_weight_stats()
        
        if max_weight > max_allowed:
            # Scale factor to bring max weight down to max_allowed
            scale = max_allowed / max_weight
            
            # Scale all weights and biases
            for i in range(self.num_inputs):
                for h in range(self.num_hidden):
                    self.weights_input_to_hidden[i][h] *= scale
            
            for h in range(self.num_hidden):
                for o in range(self.num_outputs):
                    self.weights_hidden_to_output[h][o] *= scale
            
            for h in range(self.num_hidden):
                self.hidden_bias[h] *= scale
            
            for o in range(self.num_outputs):
                self.output_bias[o] *= scale
            
            return True  # Weights were normalized
        return False  # No normalization needed
    
    def force_mutation_on_dominant_neurons(self, threshold=3.0):
        """
        If any neuron has weights that are too strong compared to others,
        force mutate those specific weights.
        """
        was_forced = False
        
        # Check each hidden neuron's incoming weights
        for h in range(self.num_hidden):
            # Calculate total incoming weight strength for this hidden neuron
            incoming_strength = sum(
                abs(self.weights_input_to_hidden[i][h])
                for i in range(self.num_inputs)
            )
            incoming_strength += abs(self.hidden_bias[h])
            
            # If this neuron is too strong, force mutate its weights
            if incoming_strength > threshold * self.num_inputs:
                for i in range(self.num_inputs):
                    self.weights_input_to_hidden[i][h] = random.gauss(0, 1)
                self.hidden_bias[h] = random.gauss(0, 1)
                was_forced = True
        
        # Check each output neuron's incoming weights
        for o in range(self.num_outputs):
            incoming_strength = sum(
                abs(self.weights_hidden_to_output[h][o])
                for h in range(self.num_hidden)
            )
            incoming_strength += abs(self.output_bias[o])
            
            if incoming_strength > threshold * self.num_hidden:
                for h in range(self.num_hidden):
                    self.weights_hidden_to_output[h][o] = random.gauss(0, 1)
                self.output_bias[o] = random.gauss(0, 1)
                was_forced = True
        
        return was_forced


# ============================================================================
# CAR CLASS
# ============================================================================

class Car:
    """
    A car that drives on the track.
    
    Each car has:
    - Position and angle on the track
    - Speed and acceleration
    - Sensors to "see" the track walls
    - A neural network brain to make decisions
    """
    
    def __init__(self, start_position):
        """Create a new car at the starting position."""
        self.position = pygame.Vector2(start_position)
        self.angle = 0.0  # Facing direction in degrees
        self.velocity = 0.0  # Current speed
        
        # Movement settings
        self.max_speed = 5.0
        self.acceleration = 0.1
        self.turn_speed = 5.0  # Degrees per frame
        
        # Car appearance
        self.image = car_image
        
        # Sensor data
        self.sensor_distances = [1.0] * len(SENSOR_ANGLES)
        self.sensor_points = []  # Endpoints for drawing sensors
        
        # Status
        self.is_alive = True
        self.distance_traveled = 0.0
        self.backward_distance = 0.0  # Track backward movement separately
        self.frames_alive = 0
        
        # Circle detection
        self.position_history = []
        
        # Brain
        self.brain = None
    
    def give_brain(self, neural_network):
        """Give this car a neural network brain."""
        self.brain = neural_network
    
    def accelerate(self):
        """Speed up."""
        self.velocity = min(self.velocity + self.acceleration, self.max_speed)
    
    def brake(self):
        """Slow down (but never go backward)."""
        self.velocity = max(self.velocity - self.acceleration / 2, 0)
    
    def turn_left(self):
        """Turn the car left."""
        self.angle += self.turn_speed
    
    def turn_right(self):
        """Turn the car right."""
        self.angle -= self.turn_speed
    
    def read_sensors(self):
        """
        Use sensors to measure distance to track walls.
        
        Returns a list of normalized distances (0 = very close, 1 = far away).
        """
        # Convert angle to radians for math
        angle_radians = math.radians(self.angle)
        
        # Get the center of the car
        car_rect = self.image.get_rect(topleft=self.position)
        car_center = pygame.Vector2(car_rect.center)
        
        distances = []
        endpoints = []
        
        for relative_angle in SENSOR_ANGLES:
            # Calculate the angle of this sensor
            sensor_angle = angle_radians + math.radians(relative_angle)
            
            # Direction the sensor points
            direction = pygame.Vector2(
                math.sin(sensor_angle),
                -math.cos(sensor_angle)
            )
            
            # Cast a ray to find the wall
            distance, endpoint = ray_mask_collision(
                car_center,
                direction,
                track_mask,
                track.get_rect(),
                SENSOR_MAX_DISTANCE,
                SENSOR_STEP
            )
            
            # Normalize the distance (0 to 1)
            normalized_distance = distance / SENSOR_MAX_DISTANCE
            distances.append(normalized_distance)
            endpoints.append(endpoint)
        
        self.sensor_distances = distances
        self.sensor_points = endpoints
        
        # Check if we hit a wall
        if any(d == 0 for d in distances):
            self.is_alive = False
        
        return distances
    
    def check_collision(self, new_position):
        """Check if the car would collide at a new position."""
        # Rotate the car image
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        
        # Get the rectangle for the rotated image
        car_rect = self.image.get_rect(topleft=new_position)
        rotated_rect = rotated_image.get_rect(center=car_rect.center)
        
        # Create a mask for the rotated car
        car_mask = pygame.mask.from_surface(rotated_image)
        
        # Check for overlap with track
        overlap = track_mask.overlap(car_mask, rotated_rect.topleft)
        
        return overlap is not None
    
    def check_for_circling(self):
        """
        Check if the car is driving in circles.
        
        If the car returns to a previous position within a short time,
        we remove the progress made during that loop.
        """
        if len(self.position_history) < MIN_FRAMES_FOR_CIRCLE:
            return False
        
        # Get current position
        current_pos = self.position_history[-1][0]
        
        # Check against older positions
        for old_pos, old_distance, old_frame in self.position_history[:-MIN_FRAMES_FOR_CIRCLE]:
            distance_to_old = current_pos.distance_to(old_pos)
            
            if distance_to_old < CIRCLE_DISTANCE_THRESHOLD:
                # We found a circle! Remove progress from that loop.
                self.distance_traveled = old_distance
                self.frames_alive = old_frame
                self.is_alive = False
                return True
        
        return False
    
    def update(self):
        """Update the car's position and state."""
        if not self.is_alive:
            return
        
        self.frames_alive += 1
        
        # Calculate movement based on angle and velocity
        angle_radians = math.radians(self.angle)
        move_x = math.sin(angle_radians) * self.velocity
        move_y = math.cos(angle_radians) * self.velocity
        
        # Calculate new position
        movement = pygame.Vector2(move_x, move_y)
        new_position = self.position - movement  # Subtract because y increases downward
        
        # Check for collision
        if self.check_collision(new_position):
            self.is_alive = False
            self.velocity = 0
        else:
            # Calculate distance moved
            distance_moved = self.position.distance_to(new_position)
            
            # Track forward vs backward movement
            # Forward movement = velocity is positive
            if self.velocity > 0:
                self.distance_traveled += distance_moved
            else:
                self.backward_distance += distance_moved
            
            self.position = new_position
            
            # Record position for circle detection
            self.position_history.append((
                self.position.copy(),
                self.distance_traveled,
                self.frames_alive
            ))
            
            # Keep history limited
            if len(self.position_history) > POSITION_HISTORY_SIZE:
                self.position_history.pop(0)
            
            # Check for circling
            self.check_for_circling()
        
        # Read sensors
        self.read_sensors()
    
    def decide_and_move(self):
        """Let the neural network decide how to move."""
        if not self.is_alive or self.brain is None:
            return
        
        # Prepare inputs for the brain
        normalized_speed = self.velocity / self.max_speed
        inputs = self.sensor_distances + [normalized_speed]
        
        # Get the brain's decision
        outputs = self.brain.think(inputs)
        
        # Choose the action with highest output
        action = outputs.index(max(outputs))
        
        # Execute the action
        # 0 = accelerate (go straight)
        # 1 = accelerate + turn left
        # 2 = accelerate + turn right
        # 3 = brake (slow down)
        if action == 0:
            self.accelerate()
        elif action == 1:
            self.accelerate()
            self.turn_left()
        elif action == 2:
            self.accelerate()
            self.turn_right()
        elif action == 3:
            self.brake()
        
        # Update position
        self.update()
    
    def calculate_fitness(self):
        """
        Calculate how well this car performed.
        
        Higher fitness = better driving.
        Backward movement is heavily penalized.
        """
        # If the car didn't go anywhere, give zero fitness
        if not self.is_alive and self.distance_traveled < 100:
            return 0.0
        
        # Calculate base fitness from forward movement
        fitness = (
            self.distance_traveled * DISTANCE_REWARD
            - self.frames_alive * TIME_PENALTY
        )
        
        # Heavily penalize backward movement (10x penalty)
        backward_penalty = self.backward_distance * 2.0
        fitness -= backward_penalty
        
        return max(fitness, 0)  # Don't allow negative fitness
    
    def draw(self, window, show_sensors=True):
        """Draw the car on the screen."""
        if not self.is_alive:
            return
        
        # Draw sensors if requested
        if show_sensors:
            car_rect = self.image.get_rect(topleft=self.position)
            car_center = car_rect.center
            for endpoint in self.sensor_points:
                pygame.draw.line(window, SENSOR_COLOR, car_center, endpoint, 1)
                pygame.draw.circle(window, SENSOR_COLOR, (int(endpoint.x), int(endpoint.y)), 3)
        
        # Draw the car
        rotate_and_draw(window, self.image, self.position, self.angle)


# ============================================================================
# POPULATION MANAGER CLASS
# ============================================================================

class Population:
    """
    Manages a population of neural networks and handles evolution.
    
    Evolution process:
    1. Run all cars and measure their fitness
    2. Select the best performers as parents
    3. Create offspring by copying and mutating parents
    4. Repeat
    """
    
    def __init__(self):
        """Create a new population."""
        self.size = POPULATION_SIZE
        self.generation = 1
        self.best_fitness_ever = 0
        self.best_fitness_history = []  # Track improvement over time
        self.stagnation_count = 0  # How many generations without improvement
        
        # Create initial random population
        self.networks = [
            NeuralNetwork(INPUT_NEURONS, HIDDEN_NEURONS, OUTPUT_NEURONS)
            for _ in range(self.size)
        ]
    
    def create_cars(self, start_position):
        """Create cars with brains from this population."""
        cars = []
        for network in self.networks:
            car = Car(start_position)
            car.give_brain(network)
            car.read_sensors()  # Initialize sensors
            cars.append(car)
        return cars
    
    def evolve(self, fitness_scores):
        """
        Create the next generation based on fitness scores.
        
        Steps:
        1. Find the best performers
        2. Scale fitness to useful values
        3. Distribute offspring proportionally
        4. Add diversity if stuck
        """
        # Track best fitness
        best_fitness = max(fitness_scores)
        self.best_fitness_history.append(best_fitness)
        
        if best_fitness > self.best_fitness_ever:
            self.best_fitness_ever = best_fitness
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        
        # Sort networks by fitness (best first)
        scored_networks = list(zip(self.networks, fitness_scores))
        scored_networks.sort(key=lambda x: x[1], reverse=True)
        
        # Select top performers as parents
        parents = scored_networks[:PARENT_COUNT]
        
        # Scale fitness values for proportional distribution
        parent_fitness_values = [fitness for _, fitness in parents]
        scaled_fitness = self._scale_fitness_values(parent_fitness_values)
        
        # Calculate how many offspring each parent gets
        total_scaled_fitness = sum(scaled_fitness)
        offspring_counts = []
        
        for i, scaled in enumerate(scaled_fitness):
            if total_scaled_fitness > 0:
                proportion = scaled / total_scaled_fitness
            else:
                proportion = 1.0 / PARENT_COUNT
            
            count = int(proportion * self.size)
            offspring_counts.append((i, count))
        
        # Handle rounding - give extra offspring to top parents
        total_allocated = sum(count for _, count in offspring_counts)
        for i in range(len(offspring_counts)):
            if total_allocated >= self.size:
                break
            offspring_counts[i] = (offspring_counts[i][0], offspring_counts[i][1] + 1)
            total_allocated += 1
        
        # Create the new generation
        new_networks = []
        
        # Create offspring from each parent
        for parent_index, offspring_count in offspring_counts:
            parent_network = parents[parent_index][0]
            
            for _ in range(offspring_count):
                if len(new_networks) >= self.size:
                    break
                
                # Copy the parent and mutate
                child = parent_network.copy()
                child.mutate(BASE_MUTATION_RATE, BASE_MUTATION_STRENGTH)
                
                # Normalize weights if they get too strong
                child.normalize_weights(max_allowed=5.0)
                
                # Force mutate dominant neurons
                child.force_mutation_on_dominant_neurons(threshold=3.0)
                
                new_networks.append(child)
        
        self.networks = new_networks[:self.size]
        self.generation += 1
    
    def _scale_fitness_values(self, fitness_values):
        """
        Scale fitness values so they're proportionally meaningful.
        
        Example:
        - If fitness values are [100, 125, 150, 175]
        - We want the bottom one (100) to have a small share
        - And the top one (175) to have a larger share
        - We scale so the differences are proportional to actual performance
        """
        if not fitness_values:
            return [1.0]
        
        # Find min and max
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)
        
        # If all values are the same, give equal proportions
        if max_fitness == min_fitness:
            return [1.0] * len(fitness_values)
        
        # Scale to 0-1 range, then add 0.1 to ensure everyone gets some offspring
        # This way, the worst parent still gets a small chance
        scaled = []
        for f in fitness_values:
            normalized = (f - min_fitness) / (max_fitness - min_fitness)
            # Add base value so even worst parent gets some offspring
            scaled.append(normalized + 0.1)
        
        return scaled


# ============================================================================
# MAIN GAME LOOP
# ============================================================================

def main():
    """Run the evolution simulation."""
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    # Starting position for cars
    start_position = (50, 300)
    
    # Create the population
    population = Population()
    
    print("\n" + "=" * 60)
    print("RACING EVOLUTION SIMULATION")
    print("=" * 60)
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Parents per generation: {PARENT_COUNT}")
    print(f"Mutation rate: {BASE_MUTATION_RATE}")
    print("=" * 60 + "\n")
    
    running = True
    
    while running:
        # Create cars for this generation
        cars = population.create_cars(start_position)
        
        # Run the generation
        frame = 0
        while frame < MAX_FRAMES_PER_GENERATION:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            # Update all cars
            frame += 1
            for car in cars:
                car.decide_and_move()
            
            # Check if all cars are dead
            alive_count = sum(1 for car in cars if car.is_alive)
            if alive_count == 0:
                break
            
            # Draw everything
            screen.fill((0, 0, 0))
            screen.blit(track, (0, 0))
            screen.blit(finish, (15, 400))
            
            for car in cars:
                car.draw(screen, show_sensors=True)
            
            # Draw info text
            info_lines = [
                f"Generation: {population.generation}",
                f"Frame: {frame} / {MAX_FRAMES_PER_GENERATION}",
                f"Alive: {alive_count} / {POPULATION_SIZE}",
                f"Best ever: {population.best_fitness_ever:.1f}"
            ]
            
            for i, line in enumerate(info_lines):
                text = font.render(line, True, (255, 255, 255))
                screen.blit(text, (10, 10 + i * 30))
            
            pygame.display.flip()
            clock.tick(60)
        
        # Generation is done - calculate fitness
        fitness_scores = [car.calculate_fitness() for car in cars]
        
        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        
        # Print generation results
        print(f"Generation {population.generation}:")
        print(f"  Best fitness: {best_fitness:.1f}")
        print(f"  Average fitness: {avg_fitness:.1f}")
        print(f"  Best ever: {population.best_fitness_ever:.1f}")
        
        # Evolve to next generation
        population.evolve(fitness_scores)
        print()
    
    pygame.quit()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
