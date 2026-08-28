#!/usr/bin/env julia
# Summarize a VLBI FITS-IDI file and its fringe-search solution window — text only, no GUI.
#
#   julia --project summary.jl <fits-idi-file>
#
# The window is set by the sampling, not by any fit: the FFT search spans ±1/(2Δ) in the conjugate
# variable and resolves structure of 1/(NΔ). So Δf sets the delay window and the bandwidth its
# resolution; Δt sets the rate window and the segment duration its resolution. Oversampling only
# refines the output grid — it moves neither. Covers the default per-IF band over the whole file,
# showing a range wherever IFs or segments differ.

using FringeHunt
using FringeHunt.VLBIFiles
using FringeHunt.DataManipulation
using FringeHunt.StructArrays
using FringeHunt.Unitful
using FringeHunt.PyFormattedStrings
using FringeHunt.Dictionaries: distinct   # selective: Dictionaries also exports `filterview`
using FringeHunt.Statistics: median
using FringeHunt: load_source, split_time_segments, calculate_timesteps, format_freq

fmt(q, unit) = f"{uconvert(unit, q):.5g}"

# One value if the collection is constant, else its extremes as a range.
function fmt_span(vals, unit)
    lo, hi = extrema(vals)
    lo == hi ? fmt(lo, unit) : "$(fmt(lo, unit)) – $(fmt(hi, unit))"
end

# The uniform time grid one segment is fitted on: its step, and the number of samples spanned.
# `records_to_visarray` lays it on `first(tns):last(tns)`, so dropped samples count too.
function segment_grid(seg)
    (; dt, tns) = calculate_timesteps(seg.datetime)
    (; dt, nsteps = length(range(extrema(tns)...)))
end

# One source's scan durations, and every (scan × baseline × single-grid segment) group it
# contributes — the same partition `compute_fringefits_all` → `_process_baseline` fits, so the
# grids reported are the fitted ones.
function source_grids(wide_table, source_id)
    scans = @p load_source(wide_table, source_id) group_vg(_.scan_id)
    durations = map(scans) do scan
        lo, hi = extrema(value(scan).datetime)
        hi - lo
    end
    grids = [segment_grid(seg)
             for scan in scans
             for bl in group_vg(r -> antenna_names(r.baseline), value(scan))
             for seg in split_time_segments(value(bl))
             if length(seg) >= 2]   # ≥2 samples to fit, as `_process_baseline` requires
    (; durations, grids)
end

function report(file; io=stdout)
    uvd = VLBI.load(VLBI.UVData, file)
    wt = uvtable_wide(uvd)

    # cross-correlations only, matching `load_source`: it drops autocorrelations and then *reduces*
    # over what is left, so a source with no cross-correlation record must never reach it
    cross = @p wt filterview(!allequal(antenna_names(_.baseline)))
    isempty(cross) && error("no cross-correlation data in $file")
    bls = @p cross.baseline map(antenna_names) distinct
    ants = @p bls flatmap(identity) distinct collect sort
    sids = @p cross.source_id distinct
    tmin, tmax = extrema(cross.datetime)

    per_source = map(sids) do sid
        (; name=sources(uvd)[sid].name, source_grids(wt, sid)...)
    end
    srcnames = @p per_source map(_.name) sort
    grids = @p per_source flatmap(_.grids) StructArray
    isempty(grids) && error("no fittable data: every scan × baseline segment has < 2 time samples")
    nscans = sum(g -> length(g.durations), per_source)
    onsource = @p per_source flatmap(_.durations) sum

    # frequency: under the default `PerIF` grouping, one IF is one fitted band
    fws = uvd.freq_windows
    dfs = map(fw -> abs(step(VLBIFiles.frequencies(fw))), fws)   # the fit's freq-axis step
    bws = map(fw -> abs(fw.width), fws)
    ifs = map(fw -> (; fw.nchan, bw=abs(fw.width)), fws)
    ifdesc = @p ifs distinct collect map("$(count(==(_), ifs)) × $(_.nchan) chan spanning $(format_freq(_.bw))")
    spans = grids.nsteps .* grids.dt                             # N·Δt, the FFT's time extent
    nlo, nhi = extrema(grids.nsteps)

    println(io, basename(file))
    println(io, "$(length(sids)) sources, $(length(bls)) baselines, $nscans scans, $(length(grids)) fitted segments")
    println(io, "$tmin – $tmax  ($(fmt(tmax - tmin, u"hr")) span, $(fmt(onsource, u"hr")) in scans)")
    println(io)
    println(io, "antennas   ", join(ants, ", "))
    println(io, "sources    ", join(srcnames, ", "))
    println(io)
    println(io, "scan spans by source")
    foreach(sort(per_source; by=src -> src.name)) do src
        println(io, "  $(src.name)  $(fmt(sum(src.durations), u"hr")) total (",
                "$(length(src.durations)) scans, $(fmt_span(src.durations, u"s")) each)")
    end
    println(io)
    println(io, "frequency/delay — per IF")
    println(io, "  IFs                 ", join(ifdesc, ", "))
    println(io, "  channel step  Δf    ", fmt_span(dfs, u"MHz"))
    println(io, "  delay window        ±", fmt_span(1 ./ (2 .* dfs), u"ns"), "   = ±1/(2Δf)")
    println(io, "  delay resolution     ", fmt_span(1 ./ bws, u"ns"), "   = 1/bandwidth")
    println(io)
    println(io, "time/rate — per scan × baseline")
    println(io, "  time step     Δt    ", fmt_span(grids.dt, u"s"))
    println(io, "  rate window         ±", fmt_span(1 ./ (2 .* grids.dt), u"mHz"), "   = ±1/(2Δt)")
    println(io, "  segment length      ", nlo == nhi ? "$nlo" : "$nlo – $nhi", " steps (",
                fmt_span(spans, u"s"), ", median ", fmt(median(spans), u"s"), ")")
    println(io, "  rate resolution     ", fmt_span(1 ./ spans, u"mHz"),
                " (median ", fmt(1 / median(spans), u"mHz"), ")   = 1/duration")
end

function main(args)
    length(args) == 1 || error("usage: julia --project $(basename(PROGRAM_FILE)) <fits-idi-file>")
    file = only(args)
    isfile(file) || error("file not found: $file")
    report(file)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
