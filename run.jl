#!/usr/bin/env julia
# Launch the FringeHunt GUI on a VLBI UV-data (FITS-IDI / UVFITS) file.
#
#   julia -t auto run.jl <fits-file>
#
# `-t auto` runs the fringe fit on a background thread so the progress bar animates and the window
# stays responsive. It also works single-threaded — the window just freezes while computing.

using FringeHunt

function main(args)
    length(args) == 1 || error("usage: julia -t auto $(basename(PROGRAM_FILE)) <fits-file>")
    file = only(args)
    isfile(file) || error("file not found: $file")
    Threads.nthreads() == 1 &&
        @warn "single-threaded: the Compute progress bar won't animate — rerun with `julia -t auto`"
    FringeHunt.interactive(file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
