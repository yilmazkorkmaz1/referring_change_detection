# Referring Change Detection in Remote Sensing Imagery

## Status

- ✅ **RCDGen and RCDNet release is done.**
- 🤗 **Pretrained weights are available:** https://huggingface.co/yilmazkorkmaz/RCDGen
- 🤗 **Pretrained weights are available:** [`yilmazkorkmaz/RCDGen`](https://huggingface.co/yilmazkorkmaz/RCDGen)
- 🤗 **RCDNet pretrained weights (trained only with real datasets):** [Google Drive folder](https://drive.google.com/drive/folders/1foXpLPz3jtaQN7l6UdlDFVSgakgm6RXP?usp=share_link) (includes `SECOND-model.safetensors` and `CNAM-CD-model.safetensors`)
- 🤗 **Synthetic datasets are available on Hugging Face:**
  - **SECOND Synthetic:** [`yilmazkorkmaz/Synthetic_RCD_1`](https://huggingface.co/datasets/yilmazkorkmaz/Synthetic_RCD_1)
  - **CNAM-CD Synthetic:** [`yilmazkorkmaz/Synthetic_RCD_2`](https://huggingface.co/datasets/yilmazkorkmaz/Synthetic_RCD_2)

## RCDGen

For details on running RCDGen, data preparation, and training/inference instructions, see:
- [`RCDGen/README.md`](RCDGen/README.md)

## RCDNet

For details on running RCDNet, data preparation, and training/evaluation instructions, see:
- [`RCDNet/README.md`](RCDNet/README.md)

## Usage and citation
You are encouraged to use and distribute this code for research and development purposes with appropriate citation:

```bibtex
@inproceedings{korkmaz2026referring,
  title     = {Referring Change Detection in Remote Sensing Imagery},
  author    = {Korkmaz, Yilmaz and Paranjape, Jay N. and de Melo, Celso M. and Patel, Vishal M.},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
