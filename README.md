# Bot

## Collision Meshes

Use <https://github.com/ZealanL/RLArenaCollisionDumper>. Place the `collision_meshes` folder in the same directory as this README.

## Visualization

Place `rlviser` in the same directory as this README to visualize training. Use v0.8.7 for now as future versions use flatbuffer-based API which is not yet supported. https://github.com/VirxEC/rlviser/releases/tag/v0.8.7

## Training

`uv run -m rlrl.main --render --seed 100 --n_proc 230`
