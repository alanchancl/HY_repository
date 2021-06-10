import cv2

def bigger(im, pointx, pointy, r):
    # I为原图像，pointx和pointy为放大中心点坐标，r为放大半径    
    # 分别得到放大区域的上下左右坐标

    left = round(pointy-r)
    right = round(pointy+r)
    top = round(pointx-r)
    bottom = round(pointx+r)
    # 放大区域面积
    space = r * r
    strength = 50    # 放大强度
    # 原图像为彩色图像，要分成RGB三个分量进行处理
    im0 = im.copy()
    # 插值算法
    for x in range(top, bottom):
        offsetx = x-pointx
        for y in range(left, right):
            offsety = y-pointy
            xy = offsetx*offsetx+offsety*offsety
            if xy <= space:
                # 等比例放大
                scale = 1-xy/space
                scale = 1-strength/100*scale
                # posy和posx为放大后坐标值
                # 采用最近邻插值算法
                posy = round(offsety*scale+pointy)
                posx = round(offsetx*scale+pointx)
                im0[x, y] = im[posx, posy]
    return im0

if __name__ == "__main__":
    image = cv2.imread("20200106 _IogkZiwiA_5.jpg")
    cv2.imshow('image', image)
    
    bigger_image = bigger(image, 313, 351, 20) # 放大左眼
    bigger_image = bigger(bigger_image, 313, 479, 20)# 放大右眼
    cv2.imshow('bigger_image', bigger_image)
    cv2.waitKey(0)


