# img-compressor Presentation Outline

Use this as a 10-12 slide talk deck. Slides 1-8 form one connected explanation of JPEG from first principles to final encoding. Slides 9-12 focus on how this repository implements the idea and what the measured performance looks like.

## Slide 1 - Title

Title: JPEG Compression, DCT, and GPU Acceleration in img-compressor

Main message:
- This project shows how JPEG-style compression works end to end.
- The story starts with why JPEG exists, then follows the data through color conversion, DCT, quantization, and encoding.

Transition line:
- Start with the problem: why do we need JPEG at all?

Visual suggestion:
- Use [assets/branding/img-compressor-hero.png](../assets/branding/img-compressor-hero.png)

Speaker note:
- Frame the talk as both a JPEG explainer and a project demo.

## Slide 2 - Why JPEG Exists

Main message:
- Raw images are large because every pixel is stored directly.
- Many neighboring pixels are similar, so the file contains repeated information.
- JPEG exists to shrink photo files by removing information people are unlikely to notice.

Easy intuition:
- Think of a photo like a paragraph with repeated words and similar phrases. You do not need to store every repetition exactly to keep the message clear.

Visual suggestion:
- Show a camera photo with file size comparison before and after compression.

Transition line:
- The key question is what JPEG keeps and what it throws away.

## Slide 3 - JPEG vs PNG

Main message:
- PNG is lossless: it keeps every pixel exactly.
- JPEG is lossy: it removes detail that is less noticeable to people.
- PNG is better for screenshots, text, and logos.
- JPEG is better for photos.

Simple table idea:
- JPEG: smaller for photos, some detail loss, great for camera images.
- PNG: larger for photos, exact preservation, great for sharp graphics.

Visual suggestion:
- Side-by-side examples of the same image saved as PNG and JPEG.

Transition line:
- JPEG can afford to lose some detail because human vision is not equally sensitive to everything.

## Slide 4 - JPEG Focuses on What the Eye Notices

Main message:
- The eye is more sensitive to brightness detail than to color detail.
- JPEG uses this by preserving luminance information more carefully than chroma information.
- This is why JPEG can compress color data more aggressively without ruining the photo.

Easy intuition:
- Imagine drawing a scene in pencil for brightness, then coloring it more loosely in the background where the eye notices less.

Visual suggestion:
- Show a bright detail map versus a color-detail map on the same image.

Transition line:
- To use that idea, JPEG first changes the image from RGB into a more compression-friendly color space.

## Slide 5 - RGB Becomes YCbCr

Main message:
- JPEG converts RGB into YCbCr before compression.
- Y carries brightness.
- Cb and Cr carry color differences.
- This makes it easier to compress color more heavily than brightness.

Easy intuition:
- Instead of storing red, green, and blue as equal partners, JPEG separates brightness from color so it can protect the part people notice most.

Visual suggestion:
- Show RGB splitting into a brightness lane and two color lanes.

Transition line:
- Once brightness and color are separated, JPEG can transform the brightness-like blocks into frequency form.

## Slide 6 - What the DCT Does

Main message:
- DCT stands for Discrete Cosine Transform.
- It rewrites pixel patterns as a mix of cosine waves.
- Smooth areas use only a few low-frequency waves.
- Detailed edges need more high-frequency waves.

Easy intuition:
- Instead of storing the exact shape of a hill, DCT stores the ingredients that build the hill.

Visual suggestion:
- A wave diagram showing low-frequency and high-frequency cosines.

Transition line:
- After DCT, the image is no longer stored as raw pixels; it becomes a set of frequency values.

## Slide 7 - High Frequencies Get Reduced First

Main message:
- Most natural images have lots of smooth regions.
- After DCT, the important energy often concentrates in the top-left of the block.
- That means many high-frequency numbers can be reduced without a huge visual impact.
- The smallest detail terms are the first candidates to be weakened or removed.

Easy intuition:
- DCT turns a messy picture into a list where the first few numbers matter most.

Visual suggestion:
- Heatmap of an 8x8 DCT block with strong values near the low-frequency corner.

Transition line:
- JPEG uses a quantization matrix to decide how strongly to reduce each frequency.

## Slide 8 - Quantization Matrix and Final Encoding

Main message:
- Quantization divides DCT values by a table of numbers.
- Small details often get rounded to zero.
- Bigger details are preserved more carefully.
- This is the main lossy step in JPEG.
- The quantization matrix is larger for high-frequency coefficients and smaller for low-frequency ones.
- After quantization, JPEG encodes the result into the final file format.

Easy intuition:
- It is like reducing the precision of a ruler from millimeters to centimeters when the extra detail is not worth the space.
- It is also like saying: keep the broad shape exactly, but describe the tiny ripples more coarsely.

Visual suggestion:
- Before/after DCT coefficient grid with the quantization matrix overlaid, then a final JPEG file icon.

Closing line:
- So JPEG is really: RGB to YCbCr, split into blocks, DCT, quantize high frequencies more aggressively, then encode the result.

## Slide 9 - JPEG Pipeline End to End

Main message:
- Decode the input image.
- Split it into RGB channels.
- Process 8x8 blocks with DCT, quantization, and inverse DCT.
- Reconstruct the image and write JPEG output through libjpeg.

Why this slide matters:
- It connects the theory to the actual software pipeline in this repository.

Visual suggestion:
- Flow chart: input PNG -> planar RGB -> block transform -> reconstructed RGB -> JPEG output.

## Slide 10 - How the Code Is Organized

Main message:
- [src/main.cu](../src/main.cu) handles CLI parsing, backend selection, and reporting.
- [src/compressor_cpu.cpp](../src/compressor_cpu.cpp) contains the CPU reference path and shared quantization logic.
- [src/compressor_gpu.cu](../src/compressor_gpu.cu) contains the CUDA implementation and stream-based GPU work.
- [src/saliency.cpp](../src/saliency.cpp) computes the content-aware importance map.
- [src/jpeg_scanline_writer.cpp](../src/jpeg_scanline_writer.cpp) writes the final JPEG.

Visual suggestion:
- Screenshot-style code layout diagram or file tree.

## Slide 11 - What the Project Optimizes in Practice

Main message:
- The image is stored as planar RGB so CPU and GPU can process one channel at a time.
- The GPU path uses 3 streams and pinned memory when possible.
- The saliency map can protect visually important regions with higher quality.

Easy intuition:
- The code spends faster resources where the image matters most and saves work where the eye is less likely to notice.

Visual suggestion:
- Side-by-side saliency heatmap and image region overlays.

## Slide 12 - Measured Results

Main message:
- GPU total time: 87.102 ms
- CPU total time: 145.085 ms
- Speedup: 1.67x
- PSNR CPU vs GPU: 88.139
- Output size: GPU 200.9 KB, CPU 200.8 KB

Stage breakdown from the sample run:
- GPU saliency: 50.970 ms
- GPU upload: 1.164 ms
- GPU compute: 1.170 ms
- GPU download: 0.156 ms
- GPU encode: 72.258 ms
- CPU saliency: 62.473 ms
- CPU compute: 67.008 ms
- CPU encode: 144.967 ms

Visual suggestion:
- Bar chart comparing CPU and GPU total time, plus a small table with the numbers above.

Closing line:
- The project is a compact example of how JPEG-style math, adaptive quality, and CUDA acceleration fit together in one readable codebase.