# PNG Embeddings to PyTorch Converter

This is a standalone Python script for converting PNG image embeddings generated by [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) into PyTorch `.pt` files.

## Installation

1. Clone or download this repository to your local machine.

2. Open a terminal or command prompt and navigate to the directory containing the repository.

3. Install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. Navigate to the directory containing the script (alternatively, you can place the script in your system's `$PATH` to run it from anywhere).

2. Run the script using the following command:

```bash
python png_embedding_to_pt.py <path/to/image1.png> <path/to/image2.png> … -o <output_dir>
```
For example:

```bash
python png_embedding_to_pt.py embeddings/1.png embeddings/2.png -o pt_embeddings
```

This will save the `.pt` files to the `pt_embeddings` directory.

## How it works

This script does the following:

1. Loads the input PNG images
2. Crops out any black borders
3. Splits the least significant 4 bits of the RGB values into upper and lower data blocks, with columns of all black pixels separating them
4. XORs the upper and lower data blocks with pseudo-random blocks to reveal the original data
5. Combines the upper and lower blocks into a full data block
6. Decompresses the data block using zlib to get the raw JSON data
7. Decodes the JSON using a custom JSON decoder to convert PyTorch Tensors
8. Saves the final embedding Torch Tensors to .pt files

Code adapted from [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/textual_inversion/image_embedding.py).

## License

This script is licensed under the GNU General Public License version 3.
