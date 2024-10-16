import cv2
import numpy as np

def image_threshold():
    return np.array([
    [130, 146, 133,  95,  71,  71,  62,  78],
    [130, 146, 133,  92,  62,  71,  62,  71],
    [139, 146, 146, 120,  62,  55,  55,  55],
    [139, 139, 139, 146, 117, 112, 117, 110],
    [139, 139, 139, 139, 139, 139, 139, 139],
    [146, 142, 139, 139, 139, 143, 125, 139],
    [156, 159, 159, 159, 159, 146, 159, 159],
    [168, 159, 156, 159, 159, 159, 139, 159]
], dtype=np.uint8)

def viz(img):
    enlarge_k = 32
    cv2.imshow("test", np.kron(img, np.ones((enlarge_k, enlarge_k))))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img_v = image_threshold()
    viz(img_v)

    ## Q1
    threshold = 128
    values_less_threshold_lst = img_v[img_v < threshold]
    print(values_less_threshold_lst)    # [ 95  71  71  62  78  92  62  71  62 71 ...
                                        #   120 62  55  55  55  117 112 117 110 125 ]
    # numpy
    mean_v = values_less_threshold_lst.mean()
    std_v = values_less_threshold_lst.std()

    # mannualy mean
    total = 0
    for i in values_less_threshold_lst:
        total += i
    mean_hand_v = total / len(values_less_threshold_lst)

    # mannualy std
    import math
    std_hand_v = math.sqrt(sum((x - mean_v) ** 2 for x in values_less_threshold_lst) / len(values_less_threshold_lst))

    # validation
    assert abs(mean_hand_v - mean_v) == 0, f"Mean values do not match: {mean_hand_v} vs {mean_v}"
    assert abs(std_hand_v - std_v) == 0, f"Standard deviation values do not match: {std_hand_v} vs {std_v}"

    print(mean_v, mean_hand_v, std_v, std_hand_v)
    print("Hand-calculated mean and std match with NumPy results.")

    ## Q2
    img_v = [[0 if val < 125 else 1 for val in row] for row in img_v]
    viz(img_v)
