# Safety Checker

Safety checker is a model used in stable diffusion pipeline, aimed to identify NSFW image.
[Here](https://huggingface.co/CompVis/stable-diffusion-safety-checker) is the official description.

This project extracts its safety checker into a independent function to provide a conventional way to detect NSFW image in deep neural network.

See [test_imgs.py](https://github.com/iyume/safety-checker/blob/main/test_imgs.py) to get start.

## TODO

- add standalone safety checker implementation that depends on pytorch only. provide strength config (if possible) for NSFW detection
- add FastAPI integration
