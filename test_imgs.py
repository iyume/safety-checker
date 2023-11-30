import os

from PIL import Image

import safety_checker

for idx, filename in enumerate(sorted(os.listdir("test_imgs")), 1):
    filepath = os.path.join("test_imgs", filename)
    img = Image.open(filepath)
    res = safety_checker.run_safety_checker(img, "cuda")
    print("test", idx, filename, ":", res)

"""
test 1 1.png : False
test 2 2.jpg : True
test 3 3.jpg : False
test 4 4.jpg : False
test 5 5.jpg : True
test 6 6.jpg : False
test 7 7.jpg : True
test 8 8.jpg : True

Summary:
    wrong result on case 8
"""
