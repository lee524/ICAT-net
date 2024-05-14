import torch
from ICAT_net import ICAT_NET
import pandas as pd
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
def vis_phase_picking(
    ppk,
    spk,
    waveforms: np.ndarray,
    preds: np.ndarray,


):
    rb = 83 / 255.0
    gb = 132 / 255.0
    bb = 237 / 255.0

    # 使用归一化后的RGB值设置颜色
    custom_color_B = (rb, gb, bb)

    rr = 216 / 255.0
    gr = 80 / 255.0
    br = 64 / 255.0

    # 使用归一化后的RGB值设置颜色
    custom_color_R = (rr, gr, br)

    ry = 102 / 255.0
    gy = 20 / 255.0
    by = 102 / 255.0

    # 使用归一化后的RGB值设置颜色
    custom_color_Y = (ry, gy, by)

    rg = 25 / 255.0
    gg = 137 / 255.0
    bg = 45 / 255.0

    # 使用归一化后的RGB值设置颜色
    custom_color_G = (rg, gg, bg)

    rp = 235 / 255.0
    gp = 81 / 255.0
    bp = 156 / 255.0

    # 使用归一化后的RGB值设置颜色
    custom_color_P = (rp, gp, bp)
    # waveforms = waveforms[:, :5000]
    # preds = preds[:, :5000]

    fig, axs = plt.subplots(4, 1, figsize=(6, 5))  # 调整figsize以适应三幅图的高度

    axs[0].plot(waveforms[1], color='black', label='E')
    axs[0].axvline(ppk, color=custom_color_G, linestyle='--', label='P-wave pick', linewidth=2.0)
    axs[0].axvline(spk, color=custom_color_P, linestyle='--', label='S-wave pick', linewidth=2.0)
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].legend(loc='upper right', fontsize='x-large')
    axs[0].set_ylim([-1, 1])
    # axs[0,0].label_outer()  # 隐藏内部标签
    axs[0].tick_params(axis='both', which='major', labelsize='x-large')  # 设置坐标轴刻度的字体大小
    axs[1].plot(waveforms[2], color='black', label='N')
    axs[1].axvline(ppk, color=custom_color_G, linestyle='--', label='P-wave pick', linewidth=2.0)
    axs[1].axvline(spk, color=custom_color_P, linestyle='--', label='S-wave pick', linewidth=2.0)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].legend(loc='upper right', fontsize='x-large')
    axs[1].set_ylim([-1, 1])
    # axs[1,0].label_outer()  # 隐藏内部标签
    axs[1].tick_params(axis='both', which='major', labelsize='x-large')  # 设置坐标轴刻度的字体大小

    axs[2].plot(waveforms[0], color='black', label='Z')
    axs[2].axvline(ppk, color=custom_color_G, linestyle='--', label='P-wave pick', linewidth=2.0)
    axs[2].axvline(spk, color=custom_color_P, linestyle='--', label='S-wave pick', linewidth=2.0)
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)
    axs[2].legend(loc='upper right', fontsize='x-large')
    axs[2].set_ylim([-1, 1])
    axs[2].tick_params(axis='both', which='major', labelsize='x-large')  # 设置坐标轴刻度的字体大小
    for ax in axs[0:3]:
        ax.set_ylabel('Normalized  Amplitude', fontsize='x-large')

    axs[3].plot(preds[1], color=custom_color_R, label='Prob-P', linewidth=3.0)

    # 绘制Prob-S
    axs[3].plot(preds[2], color=custom_color_Y, label='Prob-S', linewidth=3.0)

    # 绘制Prob-D

    axs[3].axvline(ppk, color=custom_color_G, linestyle='--', label='P-wave pick', linewidth=2.0)
    axs[3].axvline(spk, color=custom_color_P, linestyle='--', label='S-wave pick', linewidth=2.0)
    axs[3].spines['top'].set_visible(False)
    axs[3].spines['right'].set_visible(False)

    # 设置y轴标签
    axs[3].set_ylabel('Probability', fontsize='x-large')

    # 设置x轴标签
    axs[3].set_xlabel('Time samples', fontsize='x-large')

    # 添加图例
    axs[3].legend(loc='upper right', fontsize='x-large')

    # 设置y轴范围
    axs[3].set_ylim([0, 1])
    axs[3].tick_params(axis='both', which='major', labelsize='x-large')  # 设置坐标轴刻度的字体大小

    # 调整布局，防止标题重叠
    plt.tight_layout()


    plt.show()

def normalize(data: np.ndarray, mode: str):

    data -= np.mean(data, axis=1, keepdims=True)

    std_data = np.std(data, axis=1, keepdims=True)
    std_data[std_data == 0] = 1
    data /= std_data
    max_data = np.max(np.abs(data), axis=1, keepdims=True)
    max_data[max_data == 0] = 1
    data /= max_data

    return data


def load_data(
    data_path: str = "./experiment/extracted_data.npy",
    event_path: str = "./experiment",

):
    # Read HDF5
    _data_dir = event_path
    filename = "sample.csv"
    meta_df = pd.read_csv(
        os.path.join(_data_dir, filename),
        low_memory=False,
    )
    for k in meta_df.columns:  ##这段代码的功能是对DataFrame中的数值型列进行缺失值填充为0，对字符串型列进行去除空格和缺失值填充为空字符串的处理。
        if meta_df[k].dtype in [np.dtype("float"), np.dtype("int")]:
            meta_df[k] = meta_df[k].fillna(0)
        elif meta_df[k].dtype in [object, np.object_, "object", "O"]:
            meta_df[k] = meta_df[k].str.replace(" ", "")
            meta_df[k] = meta_df[k].fillna("")
    idx = 0
    target_event = meta_df.iloc[idx]

    data = np.load(data_path)
    data = data[idx, :, :]

    (
        ppk,
        spk,
        mag_type,
        evmag,
        motion,
        snr_str,
    ) = itemgetter(
        "p_arrival_sample",
        "s_arrival_sample",
        "source_magnitude_type",
        "source_magnitude",  # 震级
        "p_status",
        "snr_db",
    )(
        target_event
    )

    evmag = np.clip(evmag, 0, 8,
                    dtype=np.float32)  ##震级np.clip() 函数的作用是将数组中的元素限制在指定的范围内。具体而言，它将数组中的每个元素与指定的最小值和最大值进行比较，并将超出这个范围的元素裁剪到最小值和最大值之间。
    snr_values = snr_str.strip('[]').split(".")
    # snrs = [s.strip() for s in snr_str.split(".")]

    ppk = int(ppk)
    spk = int(spk)

    event = {
        "data": data,
        "ppks": [ppk] if pd.notnull(ppk) else [],
        "spks": [spk] if pd.notnull(spk) else [],
        "emg": [evmag] if pd.notnull(evmag) else [],
        "clr": [0],  # For compatibility with other datasets
    }

    ppks_value = int(event['ppks'][0])
    spks_value = int(event['spks'][0])



    # data = np.load(data_path)
    # data = data[1, :, :]
    input_len = data.shape[-1]
    data = np.concatenate(
        (data, np.zeros((data.shape[0], 8192 - input_len))), axis=1
    )

    return data,ppks_value,spks_value


def load_checkpoint(
        save_path: str,
        device: torch.device,

):
    """Load checkpoint."""
    checkpoint = torch.load(save_path, map_location=device)

    if "model_dict" not in checkpoint:
        checkpoint = {"model_dict": checkpoint}

    checkpoint["model_dict"] = {
        k.replace("module.", "").replace("_orig_mod.", ""): v
        for k, v in checkpoint["model_dict"].items()
    }



    return checkpoint
def load_model(
    ckpt_path: str,
    device: torch.device,
    in_channels: int = 3,
):
    # Model init
    model = ICAT_NET(in_channels=in_channels)
    # Load parameters
    ckpt = load_checkpoint(ckpt_path, device=device)
    model_state_dict = ckpt["model_dict"] if "model_dict" in ckpt else ckpt
    model.load_state_dict(model_state_dict)
    model.to(device)

    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step.1 - Load Model 
    model = load_model(
        ckpt_path="./checkpoints/model.pth",
        device=device,
        in_channels=3,
    )

    # Step.2 - Load waveforms
    waveform_ndarray,ppk,spk = load_data(
        data_path="./experiment/extracted_data.npy",
        event_path="./experiment",
    )
    waveform_ndarray = waveform_ndarray[:, :8192]
    waveform_ndarray = normalize(waveform_ndarray, mode="std")
    waveform_tensor = torch.from_numpy(waveform_ndarray).reshape(1, 3, -1).to(device)
    waveform_tensor = waveform_tensor.to(torch.float32)


    # Step.3 - Inference
    preds_tensor = model(waveform_tensor)
    preds_ndarray = preds_tensor.detach().cpu().numpy().reshape(3, -1)


    # Step.4 - Visualization 
    vis_phase_picking(
        ppk=ppk,
        spk=spk,
        waveforms=waveform_ndarray,
        preds=preds_ndarray,

    )

    
