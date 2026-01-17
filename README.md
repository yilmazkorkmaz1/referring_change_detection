# Referring Change Detection in Remote Sensing Imagery

<p align="center">
  <a href="https://yilmazkorkmaz1.github.io/RCD/"><img alt="Project Page" src="https://img.shields.io/badge/Project-Page-2ea44f"></a>
  <a href="https://arxiv.org/pdf/2512.11719"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2512.11719-b31b1b"></a>
  <a href="https://huggingface.co/yilmazkorkmaz/RCDGen"><img alt="HuggingFace" src="https://img.shields.io/badge/HuggingFace-RCDGen-yellow"></a>
</p>

## Links

- **Project webpage**: [`yilmazkorkmaz1.github.io/RCD`](https://yilmazkorkmaz1.github.io/RCD/)
- **Paper (arXiv)**: [`arxiv.org/pdf/2512.11719`](https://arxiv.org/pdf/2512.11719)

## Status

- ✅ **RCDGen and RCDNet release is done.**
- 🤗 **RCDGen pretrained weights are available:** [`yilmazkorkmaz/RCDGen`](https://huggingface.co/yilmazkorkmaz/RCDGen)
- 🤗 **RCDNet pretrained weights are available (trained only with real datasets):** [Google Drive folder](https://drive.google.com/drive/folders/1foXpLPz3jtaQN7l6UdlDFVSgakgm6RXP?usp=share_link) (includes `SECOND-model.safetensors` and `CNAM-CD-model.safetensors`)
- 🤗 **Synthetic datasets are available:**
  
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
@article{korkmaz2025referring,
  title={Referring Change Detection in Remote Sensing Imagery},
  author={Korkmaz, Yilmaz and Paranjape, Jay N and de Melo, Celso M and Patel, Vishal M},
  journal={arXiv preprint arXiv:2512.11719},
  year={2025}
}
