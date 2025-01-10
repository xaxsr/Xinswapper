import onnx
import torch
from network.generator import xinswapper_generator_onnx_ready

def process_conversion(checkpoint_file, output_file):
    device = 'cuda'
    net_g = xinswapper_generator_onnx_ready(num_style_blocks=6)
    net_g.to(device)

    print(">loading saved checkpoint G...")
    checkpoint = torch.load(checkpoint_file, map_location='cpu')['model_state_dict']
    emap = checkpoint['emap']
    del checkpoint['emap']
    net_g.load_state_dict(checkpoint)
    del checkpoint

    dummy_target = torch.randn(1, 3, 128, 128).to(device)
    dummy_source = torch.randn(1, 512).to(device)

    torch.onnx.export(
        net_g,
        (dummy_target, dummy_source),
        output_file,
        export_params=True,
        opset_version=11,
        do_constant_folding=False,
        input_names=['target', 'source'],
        output_names=['output'],
    )

    print(">onnx model created... adding mapping network weights")

    model = onnx.load(output_file)

    weight_tensor = emap.data
    weight_np = weight_tensor.cpu().numpy()
    weight_onnx = onnx.helper.make_tensor('buff2fs',
                                          data_type=onnx.TensorProto.FLOAT,
                                          dims=weight_np.shape,
                                          vals=weight_np.flatten())

    model.graph.initializer.append(weight_onnx)
    onnx.save(model, output_file)

    return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', type=str, default='./checkpoints/my_run/my_run_netG.pth', help='file path to Xinswapper pytorch model')
    parser.add_argument('--output_file', type=str, default='Xinswapper.onnx', help='file path to save onnx model')

    args = parser.parse_args()

    if process_conversion(args.checkpoint_file, args.output_file):
        print(f">conversion success. onnx model saved to: '{args.output_file}'")
    else:
        print(">conversion failed.")
