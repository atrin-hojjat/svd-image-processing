from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

image_name = "cat"
image_path = "./cat.png"

image = Image.open(image_path)

# Convert the image to grayscale (black and white)
bw_image = image.convert('L')
bw_image_array = np.array(bw_image)

U, S, VT = np.linalg.svd(bw_image_array)


def compress_image(U, S, Vt, rank):
    return U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank]


compression_ratios = [1, 4, 8, 16, 32, 64, 128, 256, 512, 750, 1000, 1024, 2048]

mse_values = []
psnr_values = []

for rank in compression_ratios:
    print("Rank", rank)
    image = compress_image(U, S, VT, rank)

    mse = np.mean((image - bw_image_array) ** 2)
    psnr = 10 * (np.log10(255 / np.sqrt(mse)))
    mse_values.append(mse)
    psnr_values.append(psnr)

plt.figure(figsize=(8, 6))
plt.plot(compression_ratios, mse_values, marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
plt.title('Mean Squared Error (MSE) vs. SVD compresssion')
plt.xlabel('Rank')
plt.ylabel('MSE')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("../output", "image-comp-col-mse.jpg"))
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'text.usetex': True,
    'pgf.rcfonts': False,
    "font.family": "mononoki Nerd Font Mono",
    "font.serif": [],
    #  "font.cursive": ["mononoki Nerd Font", "mononoki Nerd Font Mono"],
})
plt.savefig(os.path.join("../output", "image-comp-col-mse.pgf"))
plt.show()

plt.figure(figsize=(8, 6))
plt.semilogx(compression_ratios, mse_values, marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
plt.title('Mean Squared Error (MSE) vs. log SVD compresssion')
plt.xlabel('Rank')
plt.ylabel('MSE')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("../output", "image-comp-col-mse-log.jpg"))
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'text.usetex': True,
    'pgf.rcfonts': False,
    "font.family": "mononoki Nerd Font Mono",
    "font.serif": [],
    #  "font.cursive": ["mononoki Nerd Font", "mononoki Nerd Font Mono"],
})
plt.savefig(os.path.join("../output", "image-comp-col-mse-log.pgf"))
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(compression_ratios, mse_values, marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
plt.title('PSNR vs. SVD compresssion')
plt.xlabel('Rank')
plt.ylabel('PSRN')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("../output", "image-comp-psnr.jpg"))
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'text.usetex': True,
    'pgf.rcfonts': False,
    "font.family": "mononoki Nerd Font Mono",
    "font.serif": [],
    #  "font.cursive": ["mononoki Nerd Font", "mononoki Nerd Font Mono"],
})
plt.savefig(os.path.join("../output", "image-comp-col-psnr.pgf"))
plt.show()


plt.figure(figsize=(8, 6))
plt.semilogx(compression_ratios, mse_values, marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
plt.title('PSNR vs. log SVD compresssion')
plt.xlabel('Rank')
plt.ylabel('PSRN')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("../output", "image-comp-col-psnr-log.jpg"))
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'text.usetex': True,
    'pgf.rcfonts': False,
    "font.family": "mononoki Nerd Font Mono",
    "font.serif": [],
    #  "font.cursive": ["mononoki Nerd Font", "mononoki Nerd Font Mono"],
})
plt.savefig(os.path.join("../output", "image-comp-col-psnr-log.pgf"))
plt.show()
