Linux build instructions
========================

Prereqs:

- `cmake` (>= 3.18)
- `build-essential` (g++, make)
- CUDA toolkit (for GPU path) or skip GPU to use CPU path
- `libjpeg-dev` for JPEG writing

Install dependencies (Debian/Ubuntu):

```bash
sudo apt update
sudo apt install -y build-essential cmake libjpeg-dev
```

Build:

```bash
./build.sh
```

Run sample:

```bash
./build/img-compressor --input tests/data/img-test.png --output tests/artifacts/out.jpg --quality 85 --compare
```

Notes:
- If CMake cannot find your CUDA toolkit automatically, set `CMAKE_CUDA_COMPILER` or install CUDA's CMake integration.
- If builds fail due to compiler mismatch, pass `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++` to the cmake command.
