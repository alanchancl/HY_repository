import cv2
import numpy as np
import os

HY_223_TO_106_MAP_LIST = list(range(0, 34)) + [(34, 35), 36, (37, 38), 39] + \
                                              [53, (52, 51), 50, (49, 48), 47] + \
                                              [117, 119, 121, 123] + \
                         list(range(132, 137)) + [61, (63, 64), (69, 70), 72, (74, 75), (80, 81)] + \
                                                 [94, (92, 91), (86, 85), 83, (102, 103),  (96, 97)] + \
                                                 [(45, 46), 44, (42, 43), 41] + \
                                                 [55, (56, 57), 58, (59, 60)] + \
                                                 [(66, 67), (77, 78), 110] + \
                                                 [(88, 89), (99, 100), 116] + \
                                                 [124, 144] + \
                                                 [130, 138, 131, 137] + \
                                                 [145, 148, 151, 153, 155, 158, 161, 206,
                                                  (203, 204), 201, (198, 199), 196] + \
                                                 [162, 166, 170, 174, 178, 190, 186, 182] + \
                                                 [109, 115]


# Check if a point is insied a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def transform_hy224_to_target(lm_224, map_list):
    lm_target = []
    for element in map_list:
        if type(element) is tuple:
            x_y_0, x_y_1 = element
            curent_xy = (lm_224[x_y_0] + lm_224[x_y_1]) / 2
        else:
            try:
                curent_xy = lm_224[element]
            except TypeError:
                return lm_target
        lm_target.append(curent_xy)
    lm_target = np.array(lm_target)
    return lm_target


def constrainPoint(p, w, h):
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
    return p


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src,
                         warpMat, (size[0], size[1]),
                         None,
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def warpTriangle(img1, img2, t1, t2):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    img2[r2[1]:r2[1] + r2[3],
         r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] +
                                     r2[2]] * (1 - mask) + img2Rect * mask


def circle_st(image, x, y, radius, color):
    x, y = y, x
    color = np.array(color)
    width, height, c = image.shape
    region_expansion = 2
    x_left = round(x - radius) - region_expansion
    x_right = round(x + radius) + region_expansion
    y_top = round(y - radius) - region_expansion
    y_bottom = round(y + radius) + region_expansion

    x_left = int(min(max(x_left, 0), width - 1))
    x_right = int(min(max(x_right, 0), width - 1))
    y_top = int(min(max(y_top, 0), height - 1))
    y_bottom = int(min(max(y_bottom, 0), height - 1))

    points = []
    ratios = []
    for i in range(x_left, x_right + 1):
        for j in range(y_top, y_bottom + 1):
            dis = ((i - x)**2 + (j - y)**2)**0.5
            if dis < radius + 0.5:
                points.append([i, j])
                ratios.append(min(radius + 0.5 - dis, 1.0))

    for idx, (xx, yy) in enumerate(points):
        image[
            xx,
            yy, :] = color * ratios[idx] + image[xx, yy, :] * (1 - ratios[idx])

    return image


def minzui_point(point, scale_lip_up, scale_lip_down):
    lip_up = [85, 86, 87, 88, 89]
    lip_down = [95, 94, 93, 92, 91]
    center_list = [(84, 96), 96, (96, 97), 97, (97, 98), 98, (98, 99), 99,
                   (99, 100), 100, (100, 90)]
    enlarge_eyes_point = point.copy()

    for _, num in enumerate(lip_up):
        length = []
        lm_target = []
        for element in center_list:
            if type(element) is tuple:
                x_y_0, x_y_1 = element
                curent_xy = (point[x_y_0] + point[x_y_1]) / 2
            else:
                curent_xy = point[element]
            length.append(sum((curent_xy - point[num])**2))
            lm_target.append(curent_xy)
        inx = length.index(min(length))
        start_point = lm_target[inx]
        pos = scale_lip_up * (point[num] - start_point) + start_point
        enlarge_eyes_point[num][1] = pos[1]
        enlarge_eyes_point[num][0] = pos[0]

    for _, num in enumerate(lip_down):
        length = []
        lm_target = []
        for element in center_list:
            if type(element) is tuple:
                x_y_0, x_y_1 = element
                curent_xy = (point[x_y_0] + point[x_y_1]) / 2
            else:
                curent_xy = point[element]
            length.append(sum((curent_xy - point[num])**2))
            lm_target.append(curent_xy)
        inx = length.index(min(length))
        start_point = lm_target[inx]
        pos = scale_lip_down * (point[num] - start_point) + start_point
        enlarge_eyes_point[num][1] = pos[1]
        enlarge_eyes_point[num][0] = pos[0]

    return enlarge_eyes_point


if __name__ == '__main__':

    url3 = "minzui_up0.3_down0.2"
    label_path = "label.txt"
    image_path = "img"

    # 读取嘴唇部分三角剖分的三角形信息
    lip_index_txt_name = "lip_index.txt"
    delaunayTri = []
    with open(lip_index_txt_name) as file:
        for line1 in file:
            x, y, z = line1.split()

            x = int(x)
            y = int(y)
            z = int(z)
            delaunayTri.append((x, y, z))

    with open(label_path) as _f:
        txt_lines = _f.readlines()
    total_num = len(txt_lines)
    current_idx = 0

    for line in txt_lines:
        current_idx += 1
        line = line.rstrip()
        line = line.rsplit('|')
        _img_path = line[0].rsplit('/')[1]
        _label = line[1]
        lm_str = _label.rsplit(" ")
        num_labels = len(lm_str)
        lm = []
        for i in range(num_labels // 2):
            lm.append((float(lm_str[i * 2 + 0]), float(lm_str[i * 2 + 1])))
        lm_target = np.array(lm)
        img_full_path = os.path.join(image_path, _img_path)
        image = cv2.imread(img_full_path)

        img_path = _img_path

        img = cv2.imread(img_full_path)

        img_orig = img.copy()

        # Rectangle to be used with Subdiv2D
        w = img.shape[1]
        h = img.shape[0]
        rect = (0, 0, w, h)
        radius = min(h, w) * (2.5 / 720)

        # 获取人工拉伸抿嘴后的点
        # scale_lip_up和scale_lip_down为上下嘴唇拉伸的幅度（0——1）
        points = lm_target
        points_Enlarge = minzui_point(points,
                                      scale_lip_up=0.3,
                                      scale_lip_down=0.2)

        img_Enlarge = np.zeros((h, w, 3), dtype=img.dtype)
        for j in range(0, len(delaunayTri)):
            tin = []
            tout = []

            for k in range(0, 3):
                pIn = points[delaunayTri[j][k]]
                pIn = constrainPoint(pIn, w, h)

                pOut = points_Enlarge[delaunayTri[j][k]]
                pOut = constrainPoint(pOut, w, h)

                tin.append(pIn)
                tout.append(pOut)

            warpTriangle(img, img_Enlarge, tin, tout)

        mask = img_Enlarge.copy()
        mask[mask > 0] = 1
        img_Enlarge = img_orig * (1 - mask) + img_Enlarge

        # # 显示
        # for landmarks_index in range(points_Enlarge.shape[0]):
        #     x_y = points_Enlarge[landmarks_index]
        #     img_Enlarge = circle_st(img_Enlarge, (x_y[0]), (x_y[1]), radius,
        #                             (0, 255, 0))
        cv2.namedWindow("img_Enlarge", 0)
        cv2.imshow("img_Enlarge", img_Enlarge)
        # cv2.imwrite(url3 + os.sep + "minzui_" + _img_path, img_Enlarge)

        cv2.waitKey(0)
