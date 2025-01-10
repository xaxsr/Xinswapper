import cv2
import torch
import numpy as np
from torchvision.transforms import v2
from skimage import transform as trans

arcface_src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],[41.5493, 92.3655], [70.7299, 92.2041]],
                       dtype=np.float32)
arcface_src = np.expand_dims(arcface_src, axis=0)

# facial alignment, taken from https://github.com/deepinsight/insightface
src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
# <--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)
# ---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)
# -->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)
# -->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)
src = np.array([src1, src2, src3, src4,src5])
src_map = {112: src, 224: src * 2}

def detect_retinaface(retinaface_model, img, max_num, score, syncvec):
    # Resize image to fit within the input_size
    input_size = (640, 640)
    im_ratio = torch.div(img.size()[1], img.size()[2])

    model_ratio = float(input_size[1]) / input_size[0]
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = torch.div(new_height, img.size()[1])

    resize = v2.Resize((new_height, new_width), antialias=True)
    img = resize(img)
    img = img.permute(1, 2, 0)

    det_img = torch.zeros((input_size[1], input_size[0], 3), dtype=torch.float32, device='cuda:0')
    det_img[:new_height, :new_width, :] = img

    # Switch to BGR and normalize
    det_img = det_img[:, :, [2, 1, 0]]
    det_img = torch.sub(det_img, 127.5)
    det_img = torch.div(det_img, 128.0)
    det_img = det_img.permute(2, 0, 1)  # 3,128,128

    # Prepare data and find model parameters
    det_img = torch.unsqueeze(det_img, 0).contiguous()

    io_binding = retinaface_model.io_binding()
    io_binding.bind_input(name='input.1', device_type='cuda', device_id=0, element_type=np.float32,
                          shape=det_img.size(), buffer_ptr=det_img.data_ptr())

    io_binding.bind_output('448', 'cuda')
    io_binding.bind_output('471', 'cuda')
    io_binding.bind_output('494', 'cuda')
    io_binding.bind_output('451', 'cuda')
    io_binding.bind_output('474', 'cuda')
    io_binding.bind_output('497', 'cuda')
    io_binding.bind_output('454', 'cuda')
    io_binding.bind_output('477', 'cuda')
    io_binding.bind_output('500', 'cuda')

    # Sync and run model
    syncvec.cpu()
    retinaface_model.run_with_iobinding(io_binding)

    net_outs = io_binding.copy_outputs_to_cpu()

    input_height = det_img.shape[2]
    input_width = det_img.shape[3]

    fmc = 3
    center_cache = {}
    scores_list = []
    bboxes_list = []
    kpss_list = []
    for idx, stride in enumerate([8, 16, 32]):
        scores = net_outs[idx]
        bbox_preds = net_outs[idx + fmc]
        bbox_preds = bbox_preds * stride

        kps_preds = net_outs[idx + fmc * 2] * stride
        height = input_height // stride
        width = input_width // stride
        K = height * width
        key = (height, width, stride)
        if key in center_cache:
            anchor_centers = center_cache[key]
        else:
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            anchor_centers = np.stack([anchor_centers] * 2, axis=1).reshape((-1, 2))
            if len(center_cache) < 100:
                center_cache[key] = anchor_centers

        pos_inds = np.where(scores >= score)[0]

        x1 = anchor_centers[:, 0] - bbox_preds[:, 0]
        y1 = anchor_centers[:, 1] - bbox_preds[:, 1]
        x2 = anchor_centers[:, 0] + bbox_preds[:, 2]
        y2 = anchor_centers[:, 1] + bbox_preds[:, 3]

        bboxes = np.stack([x1, y1, x2, y2], axis=-1)

        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)

        preds = []
        for i in range(0, kps_preds.shape[1], 2):
            px = anchor_centers[:, i % 2] + kps_preds[:, i]
            py = anchor_centers[:, i % 2 + 1] + kps_preds[:, i + 1]

            preds.append(px)
            preds.append(py)
        kpss = np.stack(preds, axis=-1)
        # kpss = kps_preds
        kpss = kpss.reshape((kpss.shape[0], -1, 2))
        pos_kpss = kpss[pos_inds]
        kpss_list.append(pos_kpss)

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]

    det_scale = det_scale.numpy()  ###

    bboxes = np.vstack(bboxes_list) / det_scale

    kpss = np.vstack(kpss_list) / det_scale
    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]

    dets = pre_det
    thresh = 0.4
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scoresb = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    orderb = scoresb.argsort()[::-1]

    keep = []
    while orderb.size > 0:
        i = orderb[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[orderb[1:]])
        yy1 = np.maximum(y1[i], y1[orderb[1:]])
        xx2 = np.minimum(x2[i], x2[orderb[1:]])
        yy2 = np.minimum(y2[i], y2[orderb[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (areas[i] + areas[orderb[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        orderb = orderb[inds + 1]

    det = pre_det[keep, :]

    kpss = kpss[order, :, :]
    kpss = kpss[keep, :, :]

    if max_num > 0 and det.shape[0] > max_num:
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                          det[:, 1])
        det_img_center = det_img.shape[0] // 2, det_img.shape[1] // 2
        offsets = np.vstack([
            (det[:, 0] + det[:, 2]) / 2 - det_img_center[1],
            (det[:, 1] + det[:, 3]) / 2 - det_img_center[0]
        ])
        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

        values = area - offset_dist_squared * 2.0  # some extra weight on the centering
        bindex = np.argsort(values)[::-1]  # some extra weight on the centering
        bindex = bindex[0:max_num]

        if kpss is not None:
            kpss = kpss[bindex, :]

    return kpss

def estimate_norm(lmk, image_size=112, mode='arcface', shrink_factor=1.0):
    global arcface_src, src_map
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    src_factor = image_size / 112
    if mode == 'arcface':
        src = arcface_src * shrink_factor + (1 - shrink_factor) * 56
        src = src * src_factor
    else:
        src = src_map[image_size] * src_factor
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index

def norm_crop(img, landmark, image_size=112, mode='arcface', shrink_factor=1.0):
    """
    Align and crop the image based of the facial landmarks in the image. The alignment is done with
    a similarity transformation based of source coordinates.
    :param img: Image to transform.
    :param landmark: Five landmark coordinates in the image.
    :param image_size: Desired output size after transformation.
    :param mode: 'arcface' aligns the face for the use of Arcface facial recognition model. Useful for
    both facial recognition tasks and face swapping tasks.
    :param shrink_factor: Shrink factor that shrinks the source landmark coordinates. This will include more border
    information around the face. Useful when you want to include more background information when performing face swaps.
    The lower the shrink factor the more of the face is included. Default value 1.0 will align the image to be ready
    for the Arcface recognition model, but usually omits part of the chin. Value of 0.0 would transform all source points
    to the middle of the image, probably rendering the alignment procedure useless.

    If you process the image with a shrink factor of 0.85 and then want to extract the identity embedding with arcface,
    you simply do a central crop of factor 0.85 to yield same cropped result as using shrink factor 1.0. This will
    reduce the resolution, the recommendation is to processed images to output resolutions higher than 112 is using
    Arcface. This will make sure no information is lost by resampling the image after central crop.
    :return: Returns the transformed image.
    """
    M, pose_index = estimate_norm(landmark, image_size, mode, shrink_factor=shrink_factor)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped
