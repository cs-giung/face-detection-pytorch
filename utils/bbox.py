import cv2


def draw_bbox(image, bbox, fill=0.0, thickness=3):

    # it will be returned
    output = image.copy()

    # fill with transparency
    if fill > 0.0:

        # fill inside bboxes
        img_fill = image.copy()
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
        img_fill = cv2.rectangle(img_fill, p1, p2, (0, 255, 0), -1)

        # overlay
        cv2.addWeighted(img_fill, fill, output, 1.0 - fill, 0, output)

    # edge with thickness
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[2]), int(bbox[3]))
    green = int(bbox[4] * 255)
    output = cv2.rectangle(output, p1, p2, (255, green, 0), thickness)

    return output


def draw_bboxes(image, bounding_boxes, fill=0.0, thickness=3):
    
    # it will be returned
    output = image.copy()

    # fill with transparency
    if fill > 0.0:

        # fill inside bboxes
        img_fill = image.copy()
        for bbox in bounding_boxes:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[2]), int(bbox[3]))
            img_fill = cv2.rectangle(img_fill, p1, p2, (0, 255, 0), -1)
        
        # overlay
        cv2.addWeighted(img_fill, fill, output, 1.0 - fill, 0, output)

    # edge with thickness
    for bbox in bounding_boxes:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
        green = int(bbox[4] * 255)
        output = cv2.rectangle(output, p1, p2, (255, green, 0), thickness)
    
    return output


def crop_thumbnail_(image, bounding_box, padding=1, size=100):

    # infos in original image
    w, h = image.shape[1], image.shape[0]
    x1, y1, x2, y2, conf = bounding_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = max(x2 - x1, y2 - y1) * padding
    
    # get thumbnail
    p1x = max(0, int(cx - r))
    p1y = max(0, int(cy - r))
    p2x = min(w, int(cx + r))
    p2y = min(h, int(cy + r))
    output = image[p1y:p2y, p1x:p2x]
    output = cv2.resize(output, (size, size), interpolation=cv2.INTER_LINEAR)

    # infos in thumbnail
    s_x = size / (p2x - p1x)
    s_y = size / (p2y - p1y)
    new_bbox = [x1 - p1x, y1 - p1y, (x2 - p1x) * s_x, (y2 - p1y) * s_y, conf]

    return output, new_bbox


def crop_thumbnail(image, bounding_box, padding=1, size=100):

    # infos in original image
    w, h = image.shape[1], image.shape[0]
    x1, y1, x2, y2, conf = bounding_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = max(x2 - x1, y2 - y1) * padding

    # get thumbnail
    p1x = int(cx - r)
    p1y = int(cy - r)
    p2x = int(cx + r)
    p2y = int(cy + r)

    img = image.copy()

    if p1x < 0:
        img = cv2.copyMakeBorder(img, top=0, bottom=0, left=-p1x, right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        x1 -= p1x
        x2 -= p1x
        p2x -= p1x
        p1x -= p1x
    if p1y < 0:
        img = cv2.copyMakeBorder(img, top=-p1y, bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        y1 -= p1y
        y2 -= p1y
        p2y -= p1y
        p1y -= p1y
    if p2x > w:
        img = cv2.copyMakeBorder(img, top=0, bottom=0, left=0, right=p2x-w, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if p2y > h:
        img = cv2.copyMakeBorder(img, top=0, bottom=p2y-h, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    output = img[p1y:p2y, p1x:p2x]
    output = cv2.resize(output, (size, size), interpolation=cv2.INTER_LINEAR)

    # infos in thumbnail
    s_x = size / (p2x - p1x)
    s_y = size / (p2y - p1y)
    new_bbox = [(x1 - p1x) * s_x, (y1 - p1y) * s_y, (x2 - p1x) * s_x, (y2 - p1y) * s_y, conf]

    return output, new_bbox
