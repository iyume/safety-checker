import os
import time

from PIL import Image

from safety_checker import SafetyChecker

model = SafetyChecker.from_pretrained_default()
model.to("cpu")

for idx, filename in enumerate(sorted(os.listdir("test_imgs")), 1):
    filepath = os.path.join("test_imgs", filename)
    img = Image.open(filepath)
    st = time.time()
    res = model.run(img)
    et = time.time()
    print(
        "test",
        idx,
        ":",
        filename,
        ":",
        str(res).ljust(5),
        ":",
        f"{os.stat(filepath).st_size // 1024}KB".ljust(6),
        ":",
        f"{et - st:.2f}s".ljust(6),
    )

"""
test 1 : 1.png : False : 169KB  : 0.54s 
test 2 : 2.jpg : True  : 589KB  : 0.49s 
test 3 : 3.jpg : False : 431KB  : 0.54s 
test 4 : 4.jpg : False : 687KB  : 0.49s 
test 5 : 5.jpg : True  : 1132KB : 0.54s 
test 6 : 6.jpg : False : 678KB  : 0.62s 
test 7 : 7.jpg : True  : 215KB  : 0.54s 
test 8 : 8.jpg : True  : 359KB  : 0.50s 

Summary:
    wrong result on case 8
"""
