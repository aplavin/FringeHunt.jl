# FringeHunt.jl 📡🔍

An interactive tool for quick fringe-finding and detection analysis of Very Long Baseline Interferometry (VLBI) datasets.

Perfect for mining archival observations for detections, and choosing which ones to proceed with for full calibration.

https://github.com/user-attachments/assets/b4199040-c464-4a89-9eec-5975da2abd8a

## Installation & Setup 🚀

1. **Clone the Repository** 📥: Download FringeHunt.jl to any location on your computer
   ```bash
   git clone https://github.com/aplavin/FringeHunt.jl.git
   cd FringeHunt.jl
   ```

2. **Install Julia** 💎: Ensure you have Julia installed: 1.10 or 1.11 is recommended

3. **Initialize the Environment** 🔧: Start Julia with the project environment
   ```bash
   julia --project
   ```

4. **Install Dependencies** 📦: In the Julia REPL, run:
   ```julia
   julia> using Pkg; Pkg.instantiate()
   ```

## Quick Start 🚀

Load FringeHunt and start exploring your VLBI data:

```julia
# Load the package
julia> using FringeHunt

# Basic Usage - Open a FITS-IDI file
julia> FringeHunt.interactive("path/to/your/data.fitsidi")

# The interactive window allows you to:
# - Select a source from the dropdown menu
# - View fringe SNR vs UV distance for all baselines
# - Click on individual fringes to inspect the raw data
```

## Built On 🏗️

FringeHunt.jl is built on top of several powerful Julia packages:

- **[VLBIFiles.jl](https://github.com/JuliaAPlavin/VLBIFiles.jl)** 📁: Reading FITS-IDI visibility files
- **[Makie.jl](https://github.com/MakieOrg/Makie.jl)** & **[MakieExtra.jl](https://github.com/JuliaAPlavin/MakieExtra.jl)** 📊: Interactive plotting and visualization
- **[DataManipulation.jl](https://github.com/JuliaAPlavin/DataManipulation.jl)** ⚡: Generic data manipulation
