import infer
import numpy as np

if __name__ == '__main__':
    try1 = infer.parse("images/cat.png")
    try2 = infer.parse("images/cat2.png")
    try3 = infer.parse("images/dog.png")

    print('infer cat.png', try1[0], try1[1])
    print('infer cat2.png', try2[0], try2[1])
    print('infer dog.png', try3[0], try3[1])

    print('try1(cat) vs try2(cat)', np.linalg.norm(try1[2] - try2[2]))
    print('try1(cat) vs try3(dog)', np.linalg.norm(try1[2] - try3[2]))
