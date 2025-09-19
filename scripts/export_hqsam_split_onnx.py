import argparse
import warnings
from typing import Dict

import torch
import torch.nn as nn

from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel

try:
    import onnxruntime  # type: ignore
    onnxruntime_exists = True
except Exception:
    onnxruntime_exists = False


def load_sam_hq(model_type: str, sam_checkpoint: str, hq_decoder_checkpoint: str) -> nn.Module:
    """
    Build a SAM-HQ model and load weights from:
      - Base SAM checkpoint (image encoder, prompt encoder, SAM mask decoder)
      - Trained HQ decoder checkpoint (MaskDecoderHQ weights from training)

    Note: The HQ decoder checkpoint typically contains only the decoder branch
    (state_dict for MaskDecoderHQ). We load it into sam.mask_decoder.
    """
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # Load HQ decoder weights into the MaskDecoderHQ inside SAM
    decoder_state: Dict[str, torch.Tensor] = torch.load(hq_decoder_checkpoint, map_location="cpu")
    missing, unexpected = sam.mask_decoder.load_state_dict(decoder_state, strict=False)
    if len(missing) > 0:
        print(f"[warn] Missing keys when loading HQ decoder: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if len(unexpected) > 0:
        print(f"[warn] Unexpected keys when loading HQ decoder: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    sam.eval()
    return sam


class SamImageEncoderOnnx(nn.Module):
    """
    Wrapper to export SAM image encoder (with normalization) to ONNX.

    Input:
      - input_image: float32 tensor in shape (1, 3, 1024, 1024), values in [0, 255]

    Outputs:
      - image_embeddings: (1, 256, 64, 64)
      - interm_embeddings: (1, 1, 64, 64, C) where C depends on model_type
    """

    def __init__(self, sam_model: nn.Module):
        super().__init__()
        self.sam = sam_model
        # Ensure buffers exist (pixel mean/std used by SAM preprocess normally)
        self.register_buffer("pixel_mean", self.sam.pixel_mean, persistent=False)
        self.register_buffer("pixel_std", self.sam.pixel_std, persistent=False)

    @torch.no_grad()
    def forward(self, input_image: torch.Tensor):
        # Expect input in range [0, 255], float32, already resized to (1024, 1024)
        x = (input_image - self.pixel_mean) / self.pixel_std
        image_embeddings, interms = self.sam.image_encoder(x)
        # Take the first global-attention block feature (as used by SAM-HQ downstream)
        first_interm = interms[0]  # (B, 64, 64, C)
        # Match the decoder ONNX expected shape index usage: [N, B, H, W, C], N=1 here
        interm_embeddings = first_interm.unsqueeze(0)
        return image_embeddings, interm_embeddings


def export_encoder_onnx(sam: nn.Module, output_path: str, opset: int, gelu_approximate: bool = False):
    model = SamImageEncoderOnnx(sam)
    if gelu_approximate:
        for _, m in model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    dummy = torch.randn(1, 3, sam.image_encoder.img_size, sam.image_encoder.img_size, dtype=torch.float)
    output_names = ["image_embeddings", "interm_embeddings"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output_path, "wb") as f:
            print(f"Exporting encoder onnx to {output_path}...")
            torch.onnx.export(
                model,
                (dummy,),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=["input_image"],
                output_names=output_names,
            )

    if onnxruntime_exists:
        providers = ["CPUExecutionProvider"]
        ort_sess = onnxruntime.InferenceSession(output_path, providers=providers)
        _ = ort_sess.run(None, {"input_image": dummy.cpu().numpy()})
        print("Encoder ONNX ran successfully with ONNXRuntime.")


def export_decoder_onnx(
    sam: nn.Module,
    output_path: str,
    model_type: str,
    opset: int,
    hq_token_only: bool = False,
    multimask_output: bool = False,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics: bool = False,
):
    onnx_model = SamOnnxModel(
        model=sam,
        hq_token_only=hq_token_only,
        multimask_output=multimask_output,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    if gelu_approximate:
        for _, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    # Build dummy inputs consistent with SAM-HQ shapes
    embed_dim = sam.prompt_encoder.embed_dim  # 256
    embed_size = sam.prompt_encoder.image_embedding_size  # (64, 64)
    encoder_embed_dim_dict = {"vit_b": 768, "vit_l": 1024, "vit_h": 1280}
    encoder_embed_dim = encoder_embed_dim_dict[model_type]

    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        # shape [N=1, B=1, H, W, C]; onnx graph indexes [0], so N=1 is fine
        "interm_embeddings": torch.randn(1, 1, *embed_size, encoder_embed_dim, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=sam.image_encoder.img_size, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=-1, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([sam.image_encoder.img_size, sam.image_encoder.img_size], dtype=torch.float),
    }

    # Trace once
    _ = onnx_model(**dummy_inputs)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    output_names = ["masks", "iou_predictions", "low_res_masks"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output_path, "wb") as f:
            print(f"Exporting decoder onnx to {output_path}...")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

    if onnxruntime_exists:
        providers = ["CPUExecutionProvider"]
        ort_inputs = {k: v.cpu().numpy() for k, v in dummy_inputs.items()}
        ort_sess = onnxruntime.InferenceSession(output_path, providers=providers)
        _ = ort_sess.run(None, ort_inputs)
        print("Decoder ONNX ran successfully with ONNXRuntime.")


def main():
    parser = argparse.ArgumentParser("Export HQ-SAM encoder and decoder to ONNX")
    parser.add_argument("--model-type", type=str, required=True, choices=["vit_b", "vit_l", "vit_h"], help="SAM backbone type")
    parser.add_argument("--sam-checkpoint", type=str, required=True, help="Path to base SAM checkpoint (e.g., sam_vit_l_0b3195.pth)")
    parser.add_argument("--hq-decoder-checkpoint", type=str, required=True, help="Path to trained HQ decoder checkpoint (e.g., epoch_45.pth)")
    parser.add_argument("--encoder-out", type=str, required=True, help="Output path for encoder ONNX")
    parser.add_argument("--decoder-out", type=str, required=True, help="Output path for decoder ONNX")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (>=11)")
    parser.add_argument("--gelu-approximate", action="store_true", help="Use tanh approximation for GELU")
    parser.add_argument("--hq-token-only", action="store_true", help="Decoder outputs HQ token mask only")
    parser.add_argument("--multimask-output", action="store_true", help="Enable multi-mask output mode in decoder")
    parser.add_argument("--use-stability-score", action="store_true", help="Replace IoU with stability score in decoder output")
    parser.add_argument("--return-extra-metrics", action="store_true", help="Return extra metrics from decoder ONNX")

    args = parser.parse_args()

    print("Loading SAM-HQ with checkpoints...")
    sam = load_sam_hq(args.model_type, args.sam_checkpoint, args.hq_decoder_checkpoint)

    export_encoder_onnx(
        sam=sam,
        output_path=args.encoder_out,
        opset=args.opset,
        gelu_approximate=args.gelu_approximate,
    )

    export_decoder_onnx(
        sam=sam,
        output_path=args.decoder_out,
        model_type=args.model_type,
        opset=args.opset,
        hq_token_only=args.hq_token_only,
        multimask_output=args.multimask_output,
        gelu_approximate=args.gelu_approximate,
        use_stability_score=args.use_stability_score,
        return_extra_metrics=args.return_extra_metrics,
    )


if __name__ == "__main__":
    main()


