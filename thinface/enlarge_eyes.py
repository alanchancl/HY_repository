import cv2
import numpy as np
import random
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


HY_106_TO_122_MAP_LIST = list(range(0,106))+[(52, 53),(53, 72),(72, 54),(54, 55),(55, 56),(56, 73),(73, 57),(57, 52)] + \
                                            [(58, 59),(59, 75),(75 ,60),(60, 61),(61, 62),(62, 76),(76, 63),(63, 58)]


#Check if a point is insied a rectangle
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


#Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):
    trangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in trangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if (rect_contains(r, pt1) and rect_contains(r, pt2)
                and rect_contains(r, pt3)):
            cv2.line(img, pt1, pt2, delaunay_color, 1)
            cv2.line(img, pt2, pt3, delaunay_color, 1)
            cv2.line(img, pt3, pt1, delaunay_color, 1)


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


def bigger_eyes_point(point, ExpandFlag, EnlargeMethod):

    scale_conner = 0.2  # 左右眼 眼角点     的放大系数
    scale = 0.3  # 左右眼 其他点     的放大系数

    left_center_x = point[74][1]
    left_center_y = point[74][0]
    right_center_x = point[77][1]
    right_center_y = point[77][0]

    enlarge_eyes_point = point.copy()
    # 122点
    if ExpandFlag == True:
        left_eyes_index = [53, 54, 56, 57, 72, 73, 52, 55] + list(range(106, 114))
        left_eyes_up_index = [53, 54, 106, 107, 108, 109, 72]
        left_eyes_down_index = [56, 57, 110, 111, 112, 113, 73]

        right_eyes_index = [59, 60, 62, 63, 75, 76, 58, 61] + list(range(114, 122))
        right_eyes_up_index = [59, 60, 114, 115, 116, 117, 75]
        right_eyes_down_index = [62, 63, 118, 119, 120, 121, 76]

        eyes_in_corner = [55, 58]
        eyes_out_corner = [52, 61]
    else:
        left_eyes_index = [53, 54, 56, 57, 72, 73, 52, 55]
        left_eyes_up_index = [53, 54, 72]
        left_eyes_down_index = [56, 57, 73]

        right_eyes_index = [59, 60, 62, 63, 75, 76, 58, 61]
        right_eyes_up_index = [59, 60, 75]
        right_eyes_down_index = [62, 63, 76]

        eyes_in_corner = [55, 58]
        eyes_out_corner = [52, 61]

    if EnlargeMethod == "circle":
        for _, num in enumerate(left_eyes_index):
            if num in left_eyes_index and num != eyes_in_corner:
                x = point[num][1]
                y = point[num][0]
                offsetx = x - left_center_x
                offsety = y - left_center_y
                posy = round(offsety * scale + y)
                posx = round(offsetx * scale + x)
                enlarge_eyes_point[num][1] = posx
                enlarge_eyes_point[num][0] = posy

        for _, num in enumerate(right_eyes_index):
            if num in right_eyes_index and num != eyes_in_corner:
                x = point[num][1]
                y = point[num][0]
                offsetx = x - right_center_x
                offsety = y - right_center_y
                posy = round(offsety * scale + y)
                posx = round(offsetx * scale + x)
                enlarge_eyes_point[num][1] = posx
                enlarge_eyes_point[num][0] = posy

        for _, num in enumerate(eyes_in_corner):
            x = point[num][1]
            y = point[num][0]

            if num in left_eyes_index:  # 则为左眼的眼角点
                offsetx = x - left_center_x
                offsety = y - left_center_y
            elif num in right_eyes_index:  # 则为右眼的眼角点
                offsetx = x - right_center_x
                offsety = y - right_center_y

            posy = round(offsety * scale_conner + y)
            posx = round(offsetx * scale_conner + x)

            enlarge_eyes_point[num][1] = posx
            enlarge_eyes_point[num][0] = posy
    elif EnlargeMethod == "vector":
        for _, num in enumerate(left_eyes_index):  # 放大眼睛后的眼角点坐标计算
            if num in left_eyes_up_index:
                pos = scale * (point[72] - point[73]) + point[num]
                enlarge_eyes_point[num][1] = pos[1]
                enlarge_eyes_point[num][0] = pos[0]
            elif num in left_eyes_down_index:
                pos = scale * (point[73] - point[72]) + point[num]
                enlarge_eyes_point[num][1] = pos[1]
                enlarge_eyes_point[num][0] = pos[0]
        for _, num in enumerate(right_eyes_index):  # 放大眼睛后的眼角点坐标计算
            if num in right_eyes_up_index:
                pos = scale * (point[75] - point[76]) + point[num]
                enlarge_eyes_point[num][1] = pos[1]
                enlarge_eyes_point[num][0] = pos[0]
            elif num in right_eyes_down_index:
                pos = scale * (point[76] - point[75]) + point[num]
                enlarge_eyes_point[num][1] = pos[1]
                enlarge_eyes_point[num][0] = pos[0]

    return enlarge_eyes_point


if __name__ == '__main__':
    #Define window names;
    win_delaunary = "Delaunay Triangulation"
    win_voronoi = "Voronoi Diagram"
    cv2.resizeWindow(win_delaunary, 400, 200)
    # cv2.namedWindow(win_delaunary, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # cv2.namedWindow(win_delaunary,0)

    #Read in the image

    label_path = "label.txt"
    image_path = "img"


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
            img = cv2.imread(img_full_path)

            #Keep a copy around
            img_orig = img.copy()

            #Rectangle to be used with Subdiv2D
            w = img.shape[1]
            h = img.shape[0]
            rect = (0, 0, w, h)

            # 计算放大眼睛后的新的landmark点信息 并在图像中显示出来
            ExpandFlag = True
            if ExpandFlag == True:
                points = transform_hy224_to_target(lm_target, HY_106_TO_122_MAP_LIST)
                eye_index_txt_name = "eye_index_V2.txt"
            else:
                eye_index_txt_name = "eye_index.txt"

            EnlargeMethod = "vector"  #  circle or vector

            points_Enlarge = bigger_eyes_point(points, ExpandFlag, EnlargeMethod)

            # for landmarks_index in range(points_Enlarge.shape[0]):
            #     x_y = points_Enlarge[landmarks_index]
            #     img2 = circle_st(img1, (x_y[0]), (x_y[1]), radius, (0, 255, 0))
            # cv2.imshow("img2", img2)
            # cv2.waitKey(0)

            delaunayTri = []
            with open(eye_index_txt_name) as file:
                for line in file:
                    x, y, z = line.split()

                    x = int(x)
                    y = int(y)
                    z = int(z)
                    delaunayTri.append((x, y, z))

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

            #Show results
            # cv2.imshow(win_delaunary,img_Enlarge)

            # for landmarks_index in range(points_Enlarge.shape[0]):
            #     x_y = points_Enlarge[landmarks_index]
            #     img_Enlarge = circle_st(img_Enlarge, (x_y[0]), (x_y[1]), radius, (0, 255, 0))
            cv2.imshow("img_Enlarge", img_Enlarge)
            # cv2.waitKey(0)

            cv2.waitKey(0)
