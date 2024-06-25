import infer
import numpy as np

if __name__ == '__main__':
    try1 = infer.parse("ui_images/input-1.png")
    try2 = infer.parse("ui_images/input-2.png")
    try3 = infer.parse("ui_images/button-1.png")

    print('infer input-1.png', try1[0], try1[1])
    print('infer input-2.png', try2[0], try2[1])
    print('infer button-1.png', try3[0], try3[1])

    print('try1(input-1.png) vs try2(input-2.png)', np.linalg.norm(try1[2] - try2[2]))
    print('try1(input-1.png) vs try3(button-1.png)', np.linalg.norm(try1[2] - try3[2]))
