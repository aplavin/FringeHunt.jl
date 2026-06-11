# FringeHunt.jl 📡🔍

An interactive tool for quick fringe-finding and detection analysis of Very Long Baseline Interferometry (VLBI) datasets.

Perfect for mining archival observations for detections, and choosing which ones to proceed with for full calibration.

https://github.com/user-attachments/assets/f467c718-9f6e-4897-b428-49b8c8baa280

## Installation & Setup 🚀

1. **Clone the Repository** 📥: Download FringeHunt.jl to any location on your computer
   ```bash
   git clone https://github.com/aplavin/FringeHunt.jl.git
   cd FringeHunt.jl
   ```

2. **Install Julia** 💎: Ensure you have Julia 1.10 installed

3. **Initialize the Environment** 🔧: Start Julia with the project environment
   ```bash
   julia --project
   ```

4. **Install Dependencies** 📦: In the Julia REPL, run:
   ```julia
   julia> using Pkg; Pkg.instantiate()
   ```

## Quick Start 🚀

Run FringeHunt on a VLBI data file (FITS-IDI / UVFITS), passing the file as the only argument:

```bash
julia --project -tauto run.jl <fits file>
```

In the interactive window you can:
- Select a source and press **Compute** to fit fringes across all baselines
- Explore the fringe SNR vs UV distance
- Click on individual points to inspect the raw data and delay/rate fringe

`FringeHunt` only loads data for a single source into memory at a time, so it can efficiently handle large files with many sources.

The future vision is to integrate `FringeHunt` with `VLBInspect.jl` in some way.

## Built On 🏗️

FringeHunt.jl is built on top of several key packages:

- **[VLBIFiles.jl](https://github.com/JuliaAPlavin/VLBIFiles.jl)** 📁: Reading FITS-IDI visibility files
- **[CImGui.jl](https://github.com/Gnimuc/CImGui.jl)** & **[ImPlot.jl](https://github.com/JuliaImGui/ImPlot.jl)** 📊: Interactive GUI and plotting (Dear ImGui + ImPlot)
- **[DataManipulation.jl](https://github.com/JuliaAPlavin/DataManipulation.jl)** ⚡: Generic data manipulation
- **[FFTW.jl](https://github.com/JuliaMath/FFTW.jl)** 🌀: Fast Fourier transforms for the fringe search
