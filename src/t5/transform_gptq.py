import argparse
import os

import torch
from diffusers import StableDiffusion3Pipeline
from modeling_t5 import T5Config, T5EncoderModel
from tqdm import tqdm
from transformers import set_seed

SUPPORTED_MODELS = ("t5",)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert weights quantized by GPTQ into AMD RyzenAI runtime format"
    )
    parser.add_argument(
        "--model-name",
        help="model name",
        type=str,
        default="t5",
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument(
        "--group-size", help="128 default", type=int, default=128, choices=[32, 128]
    )
    parser.add_argument("--gptq-weights", help="T5 GPTQ model weights", required=True)
    parser.add_argument("--op-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--preemption", action="store_true")
    parser.add_argument(
        "--pickle", action="store_true", help="Serialize the weight BOs"
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save the quantized model",
        type=str,
        default="quantized_models/",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "bfloat16", "float16"],
    )
    return parser.parse_args()


def transform_gptq(qweight, qzeros, scales, wf, bits=4):
    mask = (2**bits) - 1
    num_packs = 32 // bits

    qzeros = qzeros.unsqueeze(2)
    qzeros = qzeros.expand(-1, -1, num_packs)

    zeros = qzeros >> wf.unsqueeze(0)
    zeros = zeros.to(torch.int8)
    zeros = zeros + 1
    zeros = zeros & mask
    zeros = zeros.flatten(1)
    zeros = zeros.T

    scales = scales.T
    scales = scales.to(torch.bfloat16)

    qweight = qweight.unsqueeze(1)
    qweight = qweight.expand(-1, num_packs, -1)

    weight = qweight >> wf.unsqueeze(-1)
    weight = weight.to(torch.int8)

    weight = weight & mask
    weight = weight.flatten(0, 1).T

    weight = weight.reshape(weight.shape[0], weight.shape[1] // 2, 2)
    weight = (weight[:, :, 0] << 4) + weight[:, :, 1]

    return weight, zeros, scales


def replace_linear_node(node, qweight, qzeros, scales, bias, pickle, group_size=128):
    new_node = qlinear.QLinearPerGrp(
        model_name="t5",
        in_features=node.in_features,
        out_features=node.out_features,
        w_bit=bits,
        group_size=group_size,
        device="cpu",
    )
    new_node.weights_quantized = True
    new_node.qweight = qweight
    new_node.qzeros = qzeros
    new_node.scales = scales
    new_node.bias = bias

    if pickle:
        new_node.device = "aie"
        new_node.initialize_parameters()

    return new_node


def replace_gptq_model(model, state_dict, wf, pickle):
    for i in tqdm(range(24)):
        q_qweight_tensor = state_dict[
            f"encoder.block.{i}.layer.0.SelfAttention.q.qweight"
        ]
        q_qzeros_tensor = state_dict[
            f"encoder.block.{i}.layer.0.SelfAttention.q.qzeros"
        ]
        q_scales_tensor = state_dict[
            f"encoder.block.{i}.layer.0.SelfAttention.q.scales"
        ]
        q_bias_tensor = state_dict[f"encoder.block.{i}.layer.0.SelfAttention.q.bias"]
        qweight, qzeros, scales = transform_gptq(
            q_qweight_tensor, q_qzeros_tensor, q_scales_tensor, wf
        )
        model.encoder.block[i].layer[0].SelfAttention.q = replace_linear_node(
            model.encoder.block[i].layer[0].SelfAttention.q,
            qweight,
            qzeros,
            scales,
            q_bias_tensor,
            pickle,
        )

        k_qweight_tensor = state_dict[
            f"encoder.block.{i}.layer.0.SelfAttention.k.qweight"
        ]
        k_qzeros_tensor = state_dict[
            f"encoder.block.{i}.layer.0.SelfAttention.k.qzeros"
        ]
        k_scales_tensor = state_dict[
            f"encoder.block.{i}.layer.0.SelfAttention.k.scales"
        ]
        k_bias_tensor = state_dict[f"encoder.block.{i}.layer.0.SelfAttention.k.bias"]
        qweight, qzeros, scales = transform_gptq(
            k_qweight_tensor, k_qzeros_tensor, k_scales_tensor, wf
        )
        model.encoder.block[i].layer[0].SelfAttention.k = replace_linear_node(
            model.encoder.block[i].layer[0].SelfAttention.k,
            qweight,
            qzeros,
            scales,
            k_bias_tensor,
            pickle,
        )

        v_qweight_tensor = state_dict[
            f"encoder.block.{i}.layer.0.SelfAttention.v.qweight"
        ]
        v_qzeros_tensor = state_dict[
            f"encoder.block.{i}.layer.0.SelfAttention.v.qzeros"
        ]
        v_scales_tensor = state_dict[
            f"encoder.block.{i}.layer.0.SelfAttention.v.scales"
        ]
        v_bias_tensor = state_dict[f"encoder.block.{i}.layer.0.SelfAttention.v.bias"]
        qweight, qzeros, scales = transform_gptq(
            v_qweight_tensor, v_qzeros_tensor, v_scales_tensor, wf
        )
        model.encoder.block[i].layer[0].SelfAttention.v = replace_linear_node(
            model.encoder.block[i].layer[0].SelfAttention.v,
            qweight,
            qzeros,
            scales,
            v_bias_tensor,
            pickle,
        )

        o_qweight_tensor = state_dict[
            f"encoder.block.{i}.layer.0.SelfAttention.o.qweight"
        ]
        o_qzeros_tensor = state_dict[
            f"encoder.block.{i}.layer.0.SelfAttention.o.qzeros"
        ]
        o_scales_tensor = state_dict[
            f"encoder.block.{i}.layer.0.SelfAttention.o.scales"
        ]
        o_bias_tensor = state_dict[f"encoder.block.{i}.layer.0.SelfAttention.o.bias"]
        qweight, qzeros, scales = transform_gptq(
            o_qweight_tensor, o_qzeros_tensor, o_scales_tensor, wf
        )
        model.encoder.block[i].layer[0].SelfAttention.o = replace_linear_node(
            model.encoder.block[i].layer[0].SelfAttention.o,
            qweight,
            qzeros,
            scales,
            o_bias_tensor,
            pickle,
        )

        wi_0_qweight_tensor = state_dict[
            f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.qweight"
        ]
        wi_0_qzeros_tensor = state_dict[
            f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.qzeros"
        ]
        wi_0_scales_tensor = state_dict[
            f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.scales"
        ]
        wi_0_bias_tensor = state_dict[
            f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.bias"
        ]
        qweight, qzeros, scales = transform_gptq(
            wi_0_qweight_tensor, wi_0_qzeros_tensor, wi_0_scales_tensor, wf
        )
        model.encoder.block[i].layer[1].DenseReluDense.wi_0 = replace_linear_node(
            model.encoder.block[i].layer[1].DenseReluDense.wi_0,
            qweight,
            qzeros,
            scales,
            wi_0_bias_tensor,
            pickle,
        )

        wi_1_qweight_tensor = state_dict[
            f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.qweight"
        ]
        wi_1_qzeros_tensor = state_dict[
            f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.qzeros"
        ]
        wi_1_scales_tensor = state_dict[
            f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.scales"
        ]
        wi_1_bias_tensor = state_dict[
            f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.bias"
        ]
        qweight, qzeros, scales = transform_gptq(
            wi_1_qweight_tensor, wi_1_qzeros_tensor, wi_1_scales_tensor, wf
        )
        model.encoder.block[i].layer[1].DenseReluDense.wi_1 = replace_linear_node(
            model.encoder.block[i].layer[1].DenseReluDense.wi_1,
            qweight,
            qzeros,
            scales,
            wi_1_bias_tensor,
            pickle,
        )

        wo_weight_tensor = state_dict[
            f"encoder.block.{i}.layer.1.DenseReluDense.wo.qweight"
        ]
        wo_qzeros_tensor = state_dict[
            f"encoder.block.{i}.layer.1.DenseReluDense.wo.qzeros"
        ]
        wo_scales_tensor = state_dict[
            f"encoder.block.{i}.layer.1.DenseReluDense.wo.scales"
        ]
        wo_bias_tensor = state_dict[f"encoder.block.{i}.layer.1.DenseReluDense.wo.bias"]
        qweight, qzeros, scales = transform_gptq(
            wo_weight_tensor, wo_qzeros_tensor, wo_scales_tensor, wf
        )
        model.encoder.block[i].layer[1].DenseReluDense.wo = replace_linear_node(
            model.encoder.block[i].layer[1].DenseReluDense.wo,
            qweight,
            qzeros,
            scales,
            wo_bias_tensor,
            pickle,
        )

    return model


if __name__ == "__main__":
    set_seed(123)

    import qlinear

    args = parse_args()
    print(args)

    qmodels_dir = args.output_dir
    if not os.path.exists(qmodels_dir):
        os.makedirs(qmodels_dir)

    fname = os.path.join(
        qmodels_dir,
        f"quantized_{args.model_name}_w4_g{args.group_size}_gptq_{args.op_version}",
    )

    if args.preemption:
        fname += "_preemption"

    if args.pickle:
        fname += "_pickle"
        os.environ["NPU_WTS_CACHE"] = fname + ".bin"

    ckpt = fname + ".pth"

    torch_dtype = getattr(torch, args.dtype)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch_dtype
    )

    config_dict = pipe.text_encoder_3.config.to_dict()
    config_dict.pop("_name_or_path", None)
    config_dict["dropout_rate"] = 0.0
    t5_config = T5Config.from_dict(config_dict)

    del pipe

    gptq_state_dict = torch.load(args.gptq_weights)

    t5 = T5EncoderModel(t5_config)
    t5.load_state_dict(gptq_state_dict, strict=False)
    t5.model_name = "t5"

    t5 = t5.to(torch_dtype)

    print(t5_config)
    print(t5)

    qlinear.AIEGEMM.op_version = args.op_version
    qlinear.AIEGEMM.preemption = args.preemption
    qlinear.AIEGEMM.pickle = args.pickle
    qlinear.AIEGEMM.select_op_handle()

    bits = 4
    wf = torch.tensor(list(range(0, 32, bits)), dtype=torch.int32).unsqueeze(0)

    t5 = replace_gptq_model(
        t5,
        gptq_state_dict,
        wf,
        args.pickle,
    )

    print(t5)

    torch.save(t5, ckpt)
    print(f"Model saved in {ckpt}")
