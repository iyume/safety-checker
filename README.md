# Safety Checker

Safety checker is a model used in stable diffusion pipeline, aimed to identify NSFW image.
[Here](https://huggingface.co/CompVis/stable-diffusion-safety-checker) is the official description.

This project extracts its safety checker into a independent function to provide a conventional way to detect NSFW image in deep neural network.

## TODO

- add standalone safety checker implementation that depends on pytorch only. provide strength config (if possible) for NSFW detection
- add FastAPI integration
