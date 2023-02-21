import vision_msgs.msg as vision_msgs

def create_detection_msg(header, detections):
    """
    Create Detection2DArray ROS message.
    
    Parameters
    ----------
    header : std_msgs.Header -- header with image's timestamp
    detections : (n, 6) np.array -- n detections
                 Each detection is 2d bbox xyxy, confidence, class

    Returns
    -------
    msg : Detection2DArray
    """
    msg = vision_msgs.Detection2DArray()
    msg.header = header
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        detmsg = vision_msgs.Detection2D()
        detmsg.header = header

        # 2D bounding box
        w = int(round(x2 - x1))
        h = int(round(y2 - y1))
        cx = int(round(x1 + w / 2))
        cy = int(round(y1 + h / 2))
        detmsg.bbox.size_x = w
        detmsg.bbox.size_y = h

        detmsg.bbox.center.x = cx
        detmsg.bbox.center.y = cy

        # class id & confidence
        obj_hyp = vision_msgs.ObjectHypothesisWithPose()
        obj_hyp.id = int(cls)
        obj_hyp.score = conf
        detmsg.results = [obj_hyp]

        msg.detections.append(detmsg)

    return msg