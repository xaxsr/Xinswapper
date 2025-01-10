import os
import onnx
import torch
import numpy as np
from collections import OrderedDict

def process_conversion(onnx_file, output_file):
    if os.path.isfile(onnx_file):
        print(f">load inswapper onnx model '{onnx_file}'")
        onnx_model = onnx.load(onnx_file)
        graph_initializer = onnx_model.graph.initializer
    else:
        print(f">inswapper onnx model not found at '{onnx_file}'")
        return False

    weight_map = {
        'first_layer.1.weight': 'onnx::Conv_833',
        'first_layer.1.bias': 'onnx::Conv_834',
        'encoder1.0.weight': 'onnx::Conv_836',
        'encoder1.0.bias': 'onnx::Conv_837',
        'encoder2.0.weight': 'onnx::Conv_839',
        'encoder2.0.bias': 'onnx::Conv_840',
        'encoder3.0.weight': 'onnx::Conv_842',
        'encoder3.0.bias': 'onnx::Conv_843',
        'style_block.0.conv1.1.weight': 'styles.0.conv1.1.weight',
        'style_block.0.conv1.1.bias': 'styles.0.conv1.1.bias',
        'style_block.0.style1.weight': 'styles.0.style1.linear.weight',
        'style_block.0.style1.bias': 'styles.0.style1.linear.bias',
        'style_block.0.conv2.1.weight': 'styles.0.conv2.1.weight',
        'style_block.0.conv2.1.bias': 'styles.0.conv2.1.bias',
        'style_block.0.style2.weight': 'styles.0.style2.linear.weight',
        'style_block.0.style2.bias': 'styles.0.style2.linear.bias',
        'style_block.1.conv1.1.weight': 'styles.1.conv1.1.weight',
        'style_block.1.conv1.1.bias': 'styles.1.conv1.1.bias',
        'style_block.1.style1.weight': 'styles.1.style1.linear.weight',
        'style_block.1.style1.bias': 'styles.1.style1.linear.bias',
        'style_block.1.conv2.1.weight': 'styles.1.conv2.1.weight',
        'style_block.1.conv2.1.bias': 'styles.1.conv2.1.bias',
        'style_block.1.style2.weight': 'styles.1.style2.linear.weight',
        'style_block.1.style2.bias': 'styles.1.style2.linear.bias',
        'style_block.2.conv1.1.weight': 'styles.2.conv1.1.weight',
        'style_block.2.conv1.1.bias': 'styles.2.conv1.1.bias',
        'style_block.2.style1.weight': 'styles.2.style1.linear.weight',
        'style_block.2.style1.bias': 'styles.2.style1.linear.bias',
        'style_block.2.conv2.1.weight': 'styles.2.conv2.1.weight',
        'style_block.2.conv2.1.bias': 'styles.2.conv2.1.bias',
        'style_block.2.style2.weight': 'styles.2.style2.linear.weight',
        'style_block.2.style2.bias': 'styles.2.style2.linear.bias',
        'style_block.3.conv1.1.weight': 'styles.3.conv1.1.weight',
        'style_block.3.conv1.1.bias': 'styles.3.conv1.1.bias',
        'style_block.3.style1.weight': 'styles.3.style1.linear.weight',
        'style_block.3.style1.bias': 'styles.3.style1.linear.bias',
        'style_block.3.conv2.1.weight': 'styles.3.conv2.1.weight',
        'style_block.3.conv2.1.bias': 'styles.3.conv2.1.bias',
        'style_block.3.style2.weight': 'styles.3.style2.linear.weight',
        'style_block.3.style2.bias': 'styles.3.style2.linear.bias',
        'style_block.4.conv1.1.weight': 'styles.4.conv1.1.weight',
        'style_block.4.conv1.1.bias': 'styles.4.conv1.1.bias',
        'style_block.4.style1.weight': 'styles.4.style1.linear.weight',
        'style_block.4.style1.bias': 'styles.4.style1.linear.bias',
        'style_block.4.conv2.1.weight': 'styles.4.conv2.1.weight',
        'style_block.4.conv2.1.bias': 'styles.4.conv2.1.bias',
        'style_block.4.style2.weight': 'styles.4.style2.linear.weight',
        'style_block.4.style2.bias': 'styles.4.style2.linear.bias',
        'style_block.5.conv1.1.weight': 'styles.5.conv1.1.weight',
        'style_block.5.conv1.1.bias': 'styles.5.conv1.1.bias',
        'style_block.5.style1.weight': 'styles.5.style1.linear.weight',
        'style_block.5.style1.bias': 'styles.5.style1.linear.bias',
        'style_block.5.conv2.1.weight': 'styles.5.conv2.1.weight',
        'style_block.5.conv2.1.bias': 'styles.5.conv2.1.bias',
        'style_block.5.style2.weight': 'styles.5.style2.linear.weight',
        'style_block.5.style2.bias': 'styles.5.style2.linear.bias',
        'decoder3.1.weight': 'onnx::Conv_845',
        'decoder3.1.bias': 'onnx::Conv_846',
        'decoder2.1.weight': 'onnx::Conv_848',
        'decoder2.1.bias': 'onnx::Conv_849',
        'decoder1.0.weight': 'onnx::Conv_851',
        'decoder1.0.bias': 'onnx::Conv_852',
        'last_layer.1.weight': 'up0.1.weight',
        'last_layer.1.bias': 'up0.1.bias'
    }

    converted_state_dict = OrderedDict()
    for pytorch_key, onnx_key in weight_map.items():
        mapped = False
        for tensor in graph_initializer:
            if onnx_key == tensor.name:
                converted_state_dict[pytorch_key] = torch.from_numpy(onnx.numpy_helper.to_array(tensor).astype(np.float32))
                mapped=True
                break
        if not mapped:
            print(f">fail to map {onnx_key}->{pytorch_key}")
            return False

        print(f">mapped {onnx_key}->{pytorch_key}")

    converted_state_dict['emap'] = torch.from_numpy(onnx.numpy_helper.to_array(graph_initializer[-1]).astype(np.float32).copy())
    torch.save(converted_state_dict, output_file)
    return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_file', type=str, default='./weights/inswapper_128.onnx', help='path to input inswapper onnx model (inswapper_128.onnx)')
    parser.add_argument('--output_file', type=str, default='./weights/inswapper_128.pth', help='path to save pytorch state dict')

    args = parser.parse_args()

    if process_conversion(args.onnx_file, args.output_file):
        print(f">conversion success. weights saved to: '{args.output_file}'")
    else:
        print(f">conversion failed.")
