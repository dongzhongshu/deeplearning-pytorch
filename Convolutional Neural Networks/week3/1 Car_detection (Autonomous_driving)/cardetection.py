import argparse
import os
import matplotlib.pyplot as plt
import torchvision
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
from PIL import ImageFont, ImageDraw, Image

# from yad2k.models.keras_yolo import yolo_head
# from yad2k.utils.utils import draw_boxes, get_colors_for_classes, scale_boxes, read_classes, read_anchors, preprocess_image
import torch

'''
第一步、输入一幅图像，获取19x19x5x85的数值，也就是说把一幅图像划分为19x19的方格，每个方格中有5个box，每个box的值为(存在物体的概率，bx, by, bH, bW, (80个分类，每个分类的概率值))
第二步、对于19x19x5中5个box中每个box的概率，选出大于阈值的那些box，其余的用掩码给它过滤掉。
第三步、用Non-max Suppression算法, 幷交比(框中相交的面积/所有框的并面积），去掉那些多余的框.
'''


def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    1、
        Compute box scores by doing the elementwise product as described in Figure 4 ( 𝑝×𝑐
     ).
    The following code may help you choose the right operator:

    a = np.random.randn(19, 19, 5, 1)
    b = np.random.randn(19, 19, 5, 80)
    c = a * b # shape of c will be (19, 19, 5, 80)
    This is an example of broadcasting (multiplying vectors of different sizes).
2、
    For each box, find:

    the index of the class with the maximum box score

    the corresponding box score

    Useful References*

    tf.math.argmax
    tf.math.reduce_max
    Helpful Hints*

    For the axis parameter of argmax and reduce_max, if you want to select the last axis, one way to do so is to set axis=-1. This is similar to Python array indexing, where you can select the last position of an array using arrayname[-1].
    Applying reduce_max normally collapses the axis for which the maximum is applied. keepdims=False is the default option, and allows that dimension to be removed. You don't need to keep the last dimension after applying the maximum here.
3、
    Create a mask by using a threshold. As a reminder: ([0.9, 0.3, 0.4, 0.5, 0.1] < 0.4) returns: [False, True, False, False, True]. The mask should be True for the boxes you want to keep.
4、
    Use TensorFlow to apply the mask to box_class_scores, boxes and box_classes to filter out the boxes you don't want. You should be left with just the subset of boxes you want to keep.



    Arguments:
        boxes -- tensor of shape (19, 19, 5, 4)
        box_confidence -- tensor of shape (19, 19, 5, 1)
        box_class_probs -- tensor of shape (19, 19, 5, 80)
        threshold -- real value, if [ highest class probability score < threshold],
                     then get rid of the corresponding box

    Returns:
        scores -- tensor of shape (None,), containing the class probability score for selected boxes
        boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
        classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    x = 10
    y = torch.tensor(100)
    # Step 1: Compute box scores
    box_scores = box_class_probs * box_confidence

    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    box_classes = torch.argmax(box_scores, dim=-1)  # 获取最大值概率的索引，也就是分类值
    box_class_scores = torch.max(box_scores, dim=-1)[0]  # 获取5个box中每个Box的最大值的概率

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    #     # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold
    filtering_mask = torch.tensor(box_class_scores >= threshold)

    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    scores = torch.masked_select(box_class_scores, filtering_mask)
    boxes = torch.masked_select(boxes, filtering_mask.unsqueeze(-1)).reshape(-1, 4)
    classes = torch.masked_select(box_classes, filtering_mask)

    return scores, boxes, classes


def test_yolo_filter_boxes():
    # torch.random.seed(10)
    box_confidence = torch.randn((19, 19, 5, 1))
    boxes = torch.randn((19, 19, 5, 4))
    box_class_probs = torch.randn((19, 19, 5, 80))
    scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=0.5)
    print("scores[2] = " + str(scores[2].numpy()))
    print("boxes[2] = " + str(boxes[2].numpy()))
    print("classes[2] = " + str(classes[2].numpy()))
    print("scores.shape = " + str(scores.shape))
    print("boxes.shape = " + str(boxes.shape))
    print("classes.shape = " + str(classes.shape))

    # assert scores.shape == (1789,), "Wrong shape in scores"
    # assert boxes.shape == (1789, 4), "Wrong shape in boxes"
    # assert classes.shape == (1789,), "Wrong shape in classes"
    #
    # assert np.isclose(scores[2].numpy(), 9.270486), "Values are wrong on scores"
    # assert np.allclose(boxes[2].numpy(), [4.6399336, 3.2303846, 4.431282, -2.202031]), "Values are wrong on boxes"
    # assert classes[2].numpy() == 8, "Values are wrong on classes"

    print("\033[92m All tests passed!")


# test_yolo_filter_boxes()
def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """


    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    # Calculate the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)
    inter_width = max(0, yi2 - yi1)
    inter_height = max(0, xi2 - xi1)
    inter_area = inter_width * inter_height

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    ## (≈ 3 lines)
    box1_area = (box1_x2 - box1_x1) * ((box1_y2 - box1_y1))
    box2_area = (box2_x2 - box2_x1) * ((box2_y2 - box2_y1))
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    ## (≈ 1 line)
    iou = inter_area / union_area

    return iou


# NMS算法
    # bboxes维度为[N,4]，scores维度为[N,], 均为tensor
def nms(boxes,scores,max_boxes_tensor,iou_threshold):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    areas = (x2-x1)*(y2-y1)   # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending=True)    # 降序排列

    keep = []
    while order.numel() > 0:       # torch.numel()返回张量元素个数
        if order.numel() == 1:     # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()    # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min=x1[i])   # [N-1,]
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)   # [N-1,]

        iou = inter / (areas[i]+areas[order[1:]]-inter)  # [N-1,]
        idx = (iou <= iou_threshold).nonzero().squeeze() # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx+1]  # 修补索引之间的差值
    return torch.LongTensor(keep)   # Pytorch的索引值为LongTensor


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    # max_boxes_tensor = tf.Variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    #
    # # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    # ##(≈ 1 line)
    # nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)
    #
    # # Use tf.gather() to select only nms_indices from scores, boxes and classes
    # ##(≈ 3 lines)
    # scores = tf.gather(scores, nms_indices)
    # boxes = tf.gather(boxes, nms_indices)
    # classes = tf.gather(classes, nms_indices)
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    return scores, boxes, classes



def yolo_boxes_to_corners(box_xy, box_wh):
    """Convert YOLO box predictions to bounding box corners."""
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return torch.cat([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])


def yolo_eval(yolo_outputs, image_shape=(720, 1280), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """

    # Retrieve outputs of the YOLO model (≈1 line)
    # box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    #
    # # Convert boxes to be ready for filtering functions (convert boxes box_xy and box_wh to corner coordinates)
    # boxes = yolo_boxes_to_corners(box_xy, box_wh)
    #
    # # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
    # scores, boxes, classes = yolo_filter_boxes(boxes, box_confidence, box_class_probs, score_threshold)
    #
    # # Scale boxes back to original image shape (720, 1280 or whatever)
    # boxes = scale_boxes(boxes, image_shape)  # Network was trained to run on 608x608 images
    #
    # # Use one of the functions you've implemented to perform Non-max suppression with
    # # maximum number of boxes set to max_boxes and a threshold of iou_threshold (≈1 line)
    # scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    # return scores, boxes, classes

def predict(image_file):
    """
    Runs the graph to predict boxes for "image_file". Prints and plots the predictions.

    Arguments:
    image_file -- name of an image stored in the "images" folder.

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes

    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
    """

    # Preprocess your image
    # image, image_data = preprocess_image("images/" + image_file, model_image_size=(608, 608))
    #
    # yolo_model_outputs = yolo_model(image_data)  # It's output is of shape (m, 19, 19, 5, 85)
    # # But yolo_eval takes input a tensor contains 4 tensors: box_xy,box_wh, box_confidence & box_class_probs
    # yolo_outputs = yolo_head(yolo_model_outputs, anchors, len(class_names))
    #
    # out_scores, out_boxes, out_classes = yolo_eval(yolo_outputs, [image.size[1], image.size[0]], 10, 0.3, 0.5)
    #
    # # Print predictions info
    # print('Found {} boxes for {}'.format(len(out_boxes), "images/" + image_file))
    # # Generate colors for drawing bounding boxes.
    # colors = get_colors_for_classes(len(class_names))
    # # Draw bounding boxes on the image file
    # # draw_boxes2(image, out_scores, out_boxes, out_classes, class_names, colors, image_shape)
    # draw_boxes(image, out_boxes, out_classes, class_names, out_scores)
    # # Save the predicted bounding box on the image
    # image.save(os.path.join("out", str(image_file).split('.')[0] + "_annotated." + str(image_file).split('.')[1]),
    #            quality=100)
    # # Display the results in the notebook
    # output_image = Image.open(
    #     os.path.join("out", str(image_file).split('.')[0] + "_annotated." + str(image_file).split('.')[1]))
    # imshow(output_image)

    # return out_scores, out_boxes, out_classes