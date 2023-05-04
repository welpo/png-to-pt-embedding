#!/usr/bin/env python3

import argparse
from pathlib import Path
import json
import numpy as np
import zlib
import torch
from PIL import Image


# Custom JSON decoder for torch.Tensor.
class EmbeddingDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, d: dict) -> torch.Tensor:
        """Decode torch.Tensor from dictionary.

        Parameters
        ----------
        d : dict
            Dictionary to decode.

        Returns
        -------
        torch.Tensor
            Decoded torch.Tensor.
        """
        if "TORCHTENSOR" in d:
            return torch.from_numpy(np.array(d["TORCHTENSOR"]))
        return d


def xor_block(block: np.ndarray) -> np.ndarray:
    """XOR block with random block.

    Parameters
    ----------
    block : np.ndarray
        Block to be XORed with random block.

    Returns
    -------
    np.ndarray
        XORed block.
    """

    def lcg(m=2**32, a=1664525, c=1013904223, seed=0):
        while True:
            seed = (a * seed + c) % m
            yield seed % 255

    g = lcg()
    randblock = (
        np.array([next(g) for _ in range(np.product(block.shape))])
        .astype(np.uint8)
        .reshape(block.shape)
    )
    return np.bitwise_xor(block.astype(np.uint8), randblock & 0x0F)


def crop_black(img: np.ndarray) -> np.ndarray:
    """Crop black borders from an image.

    Parameters
    ----------
    img : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Cropped image.
    """
    # Create a boolean mask where True represents non-black pixels.
    mask = (img > 0).all(2)
    # Collapse the mask along x and y axes, respectively.
    mask0, mask1 = mask.any(0), mask.any(1)
    # Find the start and end columns of the non-black region.
    col_start, col_end = mask0.argmax(), mask.shape[1] - mask0[::-1].argmax()
    # Find the start and end rows of the non-black region.
    row_start, row_end = mask1.argmax(), mask.shape[0] - mask1[::-1].argmax()
    # Crop the image using the computed start and end indices.
    return img[row_start:row_end, col_start:col_end]


def extract_image_data_embed(image: Image.Image) -> dict:
    """Extract image data and decode it.

    Parameters
    ----------
    image : Image.Image
        Image to extract data from.

    Returns
    -------
    dict
        Extracted and decoded data.
    """
    # Crop black borders and mask the least significant 4 bits.
    outarr = (
        crop_black(
            np.array(image.convert("RGB").getdata())
            .reshape(image.size[1], image.size[0], 3)
            .astype(np.uint8)
        )
        & 0x0F
    )
    # Find the columns with no data.
    black_cols = np.where(np.sum(outarr, axis=(0, 2)) == 0)

    # Split the data into lower and upper blocks.
    data_block_lower = outarr[:, : black_cols[0].min(), :]
    data_block_upper = outarr[:, black_cols[0].max() + 1 :, :]

    # XOR the data blocks with random blocks.
    data_block_lower = xor_block(data_block_lower)
    data_block_upper = xor_block(data_block_upper)

    # Combine the upper and lower data blocks.
    data_block = (data_block_upper << 4) | (data_block_lower)
    data_block = data_block.flatten().tobytes()

    # Decompress the data block.
    data = zlib.decompress(data_block)
    return json.loads(data, cls=EmbeddingDecoder)


def main(image_paths: list[Path], output_dir: Path) -> None:
    """Convert PNG embeddings to .pt files.

    Parameters
    ----------
    image_paths : list[Path]
        List of input image files.
    output_dir : Path
        Output directory for .pt files.
    """
    for image_path in image_paths:
        image = Image.open(image_path)
        embedding = extract_image_data_embed(image)

        output_path = output_dir / (image_path.stem + ".pt")
        torch.save(embedding, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PNG stable diffusion embeddings to .pt files."
    )
    parser.add_argument(
        "images", nargs="+", type=Path, help="List of input image files."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("."),
        help="Output directory for .pt files (default = current path).",
    )
    args = parser.parse_args()

    main(args.images, args.output)
