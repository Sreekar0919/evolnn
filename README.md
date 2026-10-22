# EvolNN: cars that learn to drive by evolution

A top-down racing game where a population of neural-network "drivers" learns to complete a track with
no training data and no gradient descent. Selection and mutation alone do the work.

## How it works

- **Sensors.** Each car casts 21 rays (−180° to +180° in 18° steps, 250 px range) and measures the
  distance to the nearest wall. Those distances plus the car's speed form the network input.
- **Brain.** A small feed-forward network, written from scratch: 22 inputs → 8 hidden → 4 actions.
- **Fitness.** +0.2 per pixel driven, −0.1 per frame alive (so faster is better), and a penalty for
  driving in circles. Circling is detected from each car's recent position history.
- **Evolution.** Each generation has 40 cars. The top 8 become parents, and their children are
  copies with mutated weights (each weight has a 20% chance to change). Generations without
  improvement are tracked.

## Run

```bash
pip install -r requirements.txt
python game.py
```

## Files

| File | Purpose |
|---|---|
| `game.py` | Game loop, `NeuralNetwork`, `Car`, `Population`, tuning constants at the top |
| `rays.py` | Ray-casting against track masks and rectangles |
| `imgs/` | Car, finish line and four track layouts |

Built with Python and pygame-ce.
