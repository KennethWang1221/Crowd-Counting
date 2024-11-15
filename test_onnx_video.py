#!/usr/bin/env python3

import argparse
import cv2
import os
import numpy as np

DEBUG = False

def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    return anchor_points

# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points

def AnchorPoints(image_shape, pyramid_levels=[3, 4, 5, 6, 7], strides = 8, row=2, line=2):
    strides = [2 ** x for x in pyramid_levels]
    image_shape = np.array(image_shape) # image_shape = image.shape[2:]
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]

    all_anchor_points = np.zeros((0, 2)).astype(np.float32)
    # get reference points for each level
    for idx, p in enumerate(pyramid_levels):
        anchor_points = generate_anchor_points(2**p, row=row, line=line)
        shifted_anchor_points = shift(image_shapes[idx], strides[idx], anchor_points)
        all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)

    all_anchor_points = np.expand_dims(all_anchor_points, axis=0)

    return all_anchor_points.astype(np.float32)

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def frames_to_video(frames_folder, output_video_path, fps):
    frame_files = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith('.png')])
    if not frame_files:
        raise ValueError("No frames found in the specified folder.")
    
    # Read the first frame to get the dimensions
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video.write(frame)

    video.release()
    
def preprocess(img0, file_path):

    ori_file_name =  os.path.splitext(os.path.basename(file_path))[0] 
    img0 = cv2.resize(img0, ((1024, 640)), interpolation=cv2.INTER_AREA) 
    im = (img0 / 255.0).astype(np.float32)
    im = np.transpose(im,(2,0,1))

    return im, img0, ori_file_name

def inference(model, im_preprocess, ori_file_name, inference_input_tensor_path, inference_opt_tensor_path):
    file_name = os.path.splitext(os.path.basename(ori_file_name))[0]

    if len(im_preprocess.shape) == 3:
        im_preprocess = np.expand_dims(im_preprocess, axis=0).astype(np.float32)  # expand for batch dim
    input_name = model.get_inputs()[0].name
    
    result = model.run(None, {input_name: im_preprocess})

    if DEBUG:
        np.savetxt('{}/{}_input.tensor'.format(inference_input_tensor_path, file_name,),
            im_preprocess.reshape(-1), fmt='%.6f')

    pred_logits = result[0]
    pred_points = result[1]

    if DEBUG:
        for i in range(len(result)):
            np.savetxt('{}/{}_output_{}.tensor'.format(inference_opt_tensor_path, file_name, (i)),
                    result[i].reshape(-1), fmt='%.6f')


    return pred_logits, pred_points
    
def build_onnx_model(model_file):
    providers =  ['CPUExecutionProvider']
    import onnxruntime
    session_detect = onnxruntime.InferenceSession(model_file, providers=providers )
    return session_detect

def postprocess(pred_logits, pred_points, im, ori_file_name, image_results_path):

    batch_size = 1
    image_shape = [640,1024]
    num_anchor_points = 4
    num_classes = 2
    features_fpn_1_shape = [1,256,80,128]
    size = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    threshold = 0.5
    file_name = os.path.splitext(os.path.basename(ori_file_name))[0]
    image_path = os.path.join(image_results_path, '{}.png'.format(file_name)) 

    anchor_points_res = np.tile(AnchorPoints(image_shape, pyramid_levels=[3,]), (1, 1, 1))
    pred_points = pred_points * 100
    pred_points = pred_points.transpose(0, 2, 3, 1)
    pred_points = pred_points.reshape(pred_points.shape[0], -1, 2)
    pred_points = pred_points + anchor_points_res
    pred_logits = pred_logits.transpose(0, 2, 3, 1)
    batch_size, width, height, _ = pred_logits.shape
    pred_logits = pred_logits.reshape(batch_size, width, height, num_anchor_points, num_classes)
    pred_logits = pred_logits.reshape(features_fpn_1_shape[0], -1, num_classes)
    outputs = {
        'pred_logits': pred_logits,
        'pred_points': pred_points
    }

    outputs_scores = softmax(outputs['pred_logits'], axis=-1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]

    # filter the predictions
    points = outputs_points[outputs_scores > threshold]
    predict_cnt = int((outputs_scores > threshold).sum())

    outputs_scores = softmax(outputs['pred_logits'], axis=-1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]
    # draw the predictions
    for p in points:
        # Start of Selection
        im = cv2.circle(im, (int(p[0]), int(p[1])), size, (0, 255, 255), -1)
    # save the visualized image
    text = f'Count: {predict_cnt}'
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = im.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = im.shape[0] - 10  # 10 pixels from the bottom
    cv2.putText(im, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
    cv2.imwrite(image_path, im)
    
def main(**args):

    if not os.path.exists(args['opts_dir']):
        os.makedirs(args['opts_dir'])

    video_frames = os.path.join(args['opts_dir'], 'video_frames') 
    inference_opt_tensor_path = os.path.join(args['opts_dir'], 'inference_opt_tensor')
    image_results_path = os.path.join(args['opts_dir'], 'image_res') 
    inference_input_tensor_path = os.path.join(args['opts_dir'], 'inference_input_tensor')
    mp4_res = os.path.join(args['opts_dir'], '{}.mp4'.format(os.path.basename(args['opts_dir'])))

    if not os.path.exists(inference_opt_tensor_path):
        os.makedirs(inference_opt_tensor_path)
    if not os.path.exists(image_results_path):
        os.makedirs(image_results_path)
    if not os.path.exists(inference_input_tensor_path):
        os.makedirs(inference_input_tensor_path)
    if not os.path.exists(video_frames):
        os.makedirs(video_frames)

    model = build_onnx_model(args['model_file'])

    cap = cv2.VideoCapture(args['video_path'])
    frame_count = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(video_frames, f'frame_{frame_count:04d}.png')
        if DEBUG:
            cv2.imwrite(frame_filename, frame)

        im_preprocess, ori_image, ori_file_name = preprocess(frame, frame_filename)
        pred_logits, pred_points = inference(model, im_preprocess, ori_file_name, inference_input_tensor_path, inference_opt_tensor_path)
        postprocess(pred_logits, pred_points, ori_image, ori_file_name, image_results_path)

        frame_count += 1

    cap.release()

    frames_to_video(image_results_path, mp4_res, fps=int(frame_rate))

if  __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo')
    # Load file
    parser.add_argument("--model_file", type=str,default="./model/crowd_counting.onnx", \
                        help='path to model')
    parser.add_argument("--video_path", type=str, default="./video.mp4", \
                        help='path to video')
    parser.add_argument("--opts_dir", type=str, default="./res", \
                        help='path of outputs files ')
    argspar = parser.parse_args()    

    print("\n### Test model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))
