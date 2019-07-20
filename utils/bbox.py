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


def crop_thumbnail(image, bounding_box, padding=1, size=100):

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
