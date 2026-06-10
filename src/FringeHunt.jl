module FringeHunt

using VLBIFiles
import FITSIO
using DataManipulation
using Statistics
using CImGui
using CImGui.CSyntax
import GLFW, ModernGL
import ImPlot, ImGuiThemes, ImPlotExtra
using ProgressLogging
using Logging
using StructArrays
using Unitful
using IntervalSets
using Uncertain
using AxisKeysExtra
using RectiGrids
using Distributions
using LinearAlgebra: norm
using AccessorsExtra
using FFTW


function load_source(uvd, source_id)
    @p uvtable_wide(uvd) |>
        filter(@o _.source_id == source_id) |>              # column-aware filter: keeps `visibility` lazy
        filter(@o !allequal(antenna_names(_.baseline))) |>  # drop autocorrelations
        VLBI.add_scan_ids(VLBI.GapBasedScans(30u"s"))
end

# one (baseline, scan) group's records → (stokes, freq, time) complex visibility on a regular time grid
function records_to_visarray(recs)
    (;dt, tns) = calculate_timesteps(recs.datetime)
    ns = @p tns extrema() range(__...)
    Z = zero(recs.visibility[1])
    @p let
        ns
        map() do n
            ix = searchsorted(tns, n)
            isempty(ix) ? Z : recs.visibility[only(ix)]
        end
        stack(KeyedArray(__, time=ns .* dt))
    end
end

function compute_fringefits_all(uvd, uvdata_src)
    scans = @p uvdata_src group_vg((; _.scan_id, ants=antenna_names(_.baseline))) collect
    @withprogress name="Fitting fringes" @p scans |>
        enumerate() |>
        flatmap() do (i, scan)
            recs = value(scan)
            alldata = records_to_visarray(recs)
            stokeslist = @p axiskeys(alldata, :stokes) filter(VLBI.is_parallel_hands)
            res = @p grid(; band=uvd.freq_windows, stokes=stokeslist) vec filtermap() do (;band, stokes)
                freqs = VLBIFiles.frequencies(band)  # per-IF channel frequencies (a StepRangeLen, needed by zeropad)
                data0 = alldata(stokes=stokes)(freq=freqs)
                data = @set named_axiskeys(data0).freq = freqs
                (;fabs, value, peakloc, ntrials) = fringefit_single(data; pad_factor=2)
                (; band, stokes, key(scan)..., uv=mean(recs.uvw), value, peakloc, ntrials)
            end
            @logprogress i/length(scans)
            res
        end
end

function compute_selected_block(uvdata_src, fringe)
    freqs = VLBIFiles.frequencies(fringe.band)
    block = @p uvdata_src |>
        filter(@o _.scan_id == fringe.scan_id) |>            # column-aware: visibility stays lazy
        filter(@o antenna_names(_.baseline) == fringe.ants) |>
        records_to_visarray |>
        __(stokes=fringe.stokes) |>
        __(freq=freqs)
    @set named_axiskeys(block).freq = freqs
end

# Captures ProgressLogging fractions (0..1) into an atomic the GUI polls each frame.
# Non-progress messages pass through to `parent`, so warnings/errors are never swallowed.
mutable struct FractionLogger <: AbstractLogger
    fraction::Threads.Atomic{Float64}
    parent::AbstractLogger
end
FractionLogger(parent=current_logger()) = FractionLogger(Threads.Atomic{Float64}(0.0), parent)

Logging.min_enabled_level(l::FractionLogger) = min(ProgressLogging.ProgressLevel, Logging.min_enabled_level(l.parent))
Logging.shouldlog(l::FractionLogger, level, args...) = level == ProgressLogging.ProgressLevel || Logging.shouldlog(l.parent, level, args...)
Logging.catch_exceptions(l::FractionLogger) = Logging.catch_exceptions(l.parent)

function Logging.handle_message(l::FractionLogger, level, message, _module, group, id, file, line; kwargs...)
    progress = ProgressLogging.asprogress(level, message, _module, group, id, file, line; kwargs...)
    if !isnothing(progress)
        l.fraction[] = something(progress.fraction, l.fraction[])
        return nothing
    end
    Logging.handle_message(l.parent, level, message, _module, group, id, file, line; kwargs...)
end

# ---------------------------------------------------------------------------
# Interactive GUI (Dear ImGui + ImPlot)
# ---------------------------------------------------------------------------

# Per-fringe scalar quantities for the UV-vs-SNR scatter (computed once after a fit).
fringe_uvdist_km(fringe) = ustrip(u"km", hypot(fringe.uv[1], fringe.uv[2]))   # projected baseline length
fringe_snr(fringe) = U.nσ(fringe.value)
fringe_freq_ghz(fringe) = ustrip(u"GHz", VLBIFiles.frequency(fringe.band))

# Immutable snapshot of one completed fit, published atomically by the compute task and read by the
# render thread. Bundling avoids torn reads of separate fields mid-publish.
struct FitResult
    uvdata_src
    fringefits::Vector
    x::Vector{Float64}        # UV distance [km] per fringe
    y::Vector{Float64}        # SNR per fringe
    freq::Vector{Float64}     # frequency [GHz] per fringe
    fmin::Float64             # frequency colorbar range [GHz]
    fmax::Float64
    strongest::Int            # index of the highest-SNR fringe (initial selection)
end

# All mutable GUI state, owned by the render thread. The compute task only publishes `result` (one
# atomic reference); `running`/`fraction` are atomic. Everything else is render-thread-only.
mutable struct AppState
    const uvd
    const source_ids::Vector{Int}                 # source_id per dropdown entry, sorted by name
    const source_names::Vector{String}            # dropdown labels (parallel to source_ids)
    sel_source::Cint                              # 0-based combo index into source_ids
    pad_factor::Cint                              # FFT oversampling slider (1..10)

    @atomic result::Union{Nothing,FitResult}      # published by the compute task
    shown::Union{Nothing,FitResult}               # result the render thread has initialized for
    fr_colors::Vector{CImGui.ImU32}               # per-fringe marker color (viridis by freq)
    selected::Int                                 # index into shown.fringefits, or 0 if none

    running::Threads.Atomic{Bool}
    fl::FractionLogger
    task::Union{Nothing,Task}

    # cache for the selected-block heatmaps; recomputed only when these key inputs change
    cache_key::Any                                # (selected, pad_factor) the cache was built for
    cache_block::Any                              # abs.(block) :: KeyedArray (or nothing)
    cache_phase::Any                              # angle.(block)
    cache_fabs::Any                               # fringefit_single fabs
    cache_peak::Any                               # (; delay, rate)
    refit_axes::Bool                              # ask ImPlot to refit heatmap axes next frame
end

function AppState(uvd)
    srcs = sources(uvd)
    order = sortperm([v.name for v in values(srcs)])
    ids = collect(keys(srcs))[order]
    names = [srcs[i].name for i in ids]
    AppState(uvd, ids, names, Cint(0), Cint(4),
             nothing, nothing, CImGui.ImU32[], 0,
             Threads.Atomic{Bool}(false), FractionLogger(), nothing,
             nothing, nothing, nothing, nothing, nothing, false)
end

# Kick off the heavy fit on a background thread. The task does ONLY compute and publishes its
# results under `app.lock`; it never touches any ImGui/ImPlot/GL state (that stays on the render
# thread). The render loop polls `app.running` / `app.fl.fraction` each frame for the progress bar.
function start_compute!(app::AppState)
    app.running[] && return
    source_id = app.source_ids[app.sel_source + 1]
    app.fl.fraction[] = 0.0
    app.running[] = true
    app.task = Threads.@spawn begin
        try
            src, ffs = Logging.with_logger(app.fl) do
                s = load_source(app.uvd, source_id)
                (s, compute_fringefits_all(app.uvd, s))
            end
            # pure DATA only — NO ImPlot/GL calls here. Per-fringe marker colors need the live ImPlot
            # context (ImPlot.SampleColormap), so they are built on the render thread (see ensure_colors!).
            xs = map(fringe_uvdist_km, ffs)
            ys = map(fringe_snr, ffs)
            fs = map(fringe_freq_ghz, ffs)
            fmin, fmax = isempty(fs) ? (0.0, 1.0) : extrema(fs)
            strongest = isempty(ffs) ? 0 : argmax(ys)
            @atomic app.result = FitResult(src, ffs, xs, ys, fs, fmin, fmax, strongest)
        catch e
            @error "fringe computation failed" exception=(e, catch_backtrace())
        finally
            app.running[] = false
        end
    end
    nothing
end

# Pick up a freshly-published result (one atomic read). On a new result, reset the render-thread
# state derived from it: selection, marker colors, and the heatmap cache.
function sync_result!(app::AppState)
    r = @atomic app.result
    if r !== app.shown
        app.shown = r
        app.selected = isnothing(r) ? 0 : r.strongest
        app.fr_colors = CImGui.ImU32[]
        app.cache_key = nothing
    end
    nothing
end

# Recompute the selected-block heatmaps if (selection, pad_factor) changed since last build.
# Runs on the render thread (fast); caches the three displayed matrices on `app`.
function ensure_block_cache!(app::AppState)
    r = app.shown
    (isnothing(r) || app.selected == 0) && return
    key = (app.selected, app.pad_factor)
    app.cache_key == key && return
    fringe = r.fringefits[app.selected]
    block = compute_selected_block(r.uvdata_src, fringe)
    fab = fringefit_single(block; pad_factor=Int(app.pad_factor))
    app.cache_block = abs.(block)
    app.cache_phase = angle.(block)
    app.cache_fabs = fab.fabs
    app.cache_peak = fab.peakloc
    app.cache_key = key
    app.refit_axes = true
    nothing
end

# Build the per-fringe viridis marker colors. MUST run on the render thread: ImPlot.SampleColormap
# needs the live ImPlot context (segfaults otherwise). Rebuilt only when the fringe set changes.
function ensure_colors!(app::AppState)
    r = app.shown
    (isnothing(r) || length(app.fr_colors) == length(r.x)) && return
    app.fr_colors = map(r.freq) do f
        t = r.fmax > r.fmin ? (f - r.fmin) / (r.fmax - r.fmin) : 0.5
        CImGui.ColorConvertFloat4ToU32(ImPlot.SampleColormap(Cfloat(t), ImPlot.ImPlotColormap_Viridis))
    end
    nothing
end

# Closed interval spanning the first..last axis key (unit-stripped), for image! extents.
_axis_interval(ax, unit) = (v = ustrip.(unit, ax); minimum(v) .. maximum(v))

function draw_controls!(app::AppState)
    CImGui.Begin("Controls")
    sel = Ref(app.sel_source)
    CImGui.Combo("Source", sel, app.source_names, length(app.source_names))
    app.sel_source = sel[]

    pf = Ref(app.pad_factor)
    CImGui.SliderInt("FFT oversampling", pf, Cint(1), Cint(10))
    app.pad_factor = pf[]

    running = app.running[]
    running && CImGui.BeginDisabled()
    CImGui.Button("Compute") && start_compute!(app)
    running && CImGui.EndDisabled()

    if running
        CImGui.SameLine()
        CImGui.ProgressBar(Cfloat(app.fl.fraction[]), CImGui.ImVec2(-1, 0))
    elseif (r = app.shown) !== nothing
        CImGui.Text("$(length(r.fringefits)) fringes fitted")
        if app.selected != 0
            f = r.fringefits[app.selected]
            CImGui.Text("Selected: $(f.ants[1])-$(f.ants[2]) $(f.stokes) scan $(f.scan_id)")
            CImGui.Text(string("SNR = ", round(fringe_snr(f); digits=1),
                               ", UV = ", round(fringe_uvdist_km(f); digits=0), " km"))
        end
    end
    CImGui.End()
end

function draw_uvsnr!(app::AppState)
    CImGui.Begin("UV vs SNR")
    ensure_colors!(app)
    r = app.shown
    if isnothing(r)
        CImGui.Text("Press Compute to fit fringes.")
    elseif ImPlot.BeginPlot("##uvsnr", "UV distance (km)", "SNR", CImGui.ImVec2(-1, -1))
        ImPlot.SetupAxisScale(ImPlot.ImAxis_Y1, ImPlot.ImPlotScale_SymLog)
        colors = app.fr_colors
        GC.@preserve colors begin
            spec = ImPlot.ImPlotSpec(Marker=ImPlot.ImPlotMarker_Circle, MarkerFillColors=pointer(colors))
            ImPlot.PlotScatter("fringes", r.x, r.y; spec)
        end
        # highlight the selected fringe with a larger distinct marker
        if app.selected != 0
            ImPlot.PlotScatter("selected", [r.x[app.selected]], [r.y[app.selected]];
                spec=ImPlot.ImPlotSpec(Marker=ImPlot.ImPlotMarker_Circle, MarkerSize=9,
                    MarkerFillColor=CImGui.ImVec4(1, 0, 0, 0.0), MarkerLineColor=CImGui.ImVec4(1, 0, 0, 1)))
        end
        # click-to-select: nearest fringe in PIXEL space (x is km, y is SymLog ⇒ data-space distance is meaningless)
        if ImPlot.IsPlotHovered() && CImGui.IsMouseClicked(0) && !isempty(r.x)
            mp = ImPlot.GetPlotMousePos()
            mpx = ImPlot.PlotToPixels(mp.x, mp.y)
            app.selected = argmin(eachindex(r.x)) do i
                px = ImPlot.PlotToPixels(r.x[i], r.y[i])
                (px.x - mpx.x)^2 + (px.y - mpx.y)^2
            end
        end
        ImPlot.EndPlot()
        CImGui.SameLine()
        ImPlot.ColormapScale("Frequency (GHz)", r.fmin, r.fmax,
            CImGui.ImVec2(80, -1), "%g", 0, ImPlot.ImPlotColormap_Viridis)
    end
    CImGui.End()
end

function draw_heatmaps!(app::AppState)
    CImGui.Begin("Heatmaps")
    if app.cache_key === nothing
        CImGui.Text("Select a fringe in the UV vs SNR plot.")
        CImGui.End()
        return
    end
    refit = app.refit_axes
    app.refit_axes = false

    fabs = app.cache_fabs
    dx = _axis_interval(axiskeys(fabs, :delay), u"ns")
    dy = _axis_interval(axiskeys(fabs, :rate), u"mHz")
    peak_d = ustrip(u"ns", app.cache_peak.delay)
    peak_r = ustrip(u"mHz", app.cache_peak.rate)

    afit = ImPlot.ImPlotAxisFlags_AutoFit
    avail = CImGui.GetContentRegionAvail()
    top_h = avail.y * 0.55f0

    refit && ImPlot.SetNextAxesToFit()
    if ImPlot.BeginPlot("Fringe Amplitude", "delay (ns)", "rate (mHz)", CImGui.ImVec2(-1, top_h);
                        x_flags=afit, y_flags=afit)
        ImPlotExtra.image!("fringe", dx, dy, fabs; colormap=:viridis, colorscale=log10)
        ImPlot.PlotInfLines("##peakd", [peak_d])
        ImPlot.PlotInfLines("##peakr", [peak_r]; spec=ImPlot.ImPlotSpec(Flags=ImPlot.ImPlotInfLinesFlags_Horizontal))
        ImPlot.EndPlot()
    end

    block = app.cache_block
    phase = app.cache_phase
    fx = _axis_interval(axiskeys(block, :freq), u"GHz")
    ty = _axis_interval(axiskeys(block, :time), u"s")
    half_w = CImGui.GetContentRegionAvail().x * 0.5f0 - 4

    refit && ImPlot.SetNextAxesToFit()
    if ImPlot.BeginPlot("Data Amplitude", "freq (GHz)", "time (s)", CImGui.ImVec2(half_w, -1);
                        x_flags=afit, y_flags=afit)
        ImPlotExtra.image!("amp", fx, ty, block; colormap=:viridis)
        ImPlot.EndPlot()
    end
    CImGui.SameLine()
    refit && ImPlot.SetNextAxesToFit()
    if ImPlot.BeginPlot("Data Phase", "freq (GHz)", "time (s)", CImGui.ImVec2(-1, -1);
                        x_flags=afit, y_flags=afit)
        ImPlotExtra.image!("phase", fx, ty, phase; colormap=:twilight,
                           colorrange=(-Float32(pi), Float32(pi)))
        ImPlot.EndPlot()
    end
    CImGui.End()
end

"""
    interactive(file)

Open the FringeHunt GUI for the UV-data `file`: pick a source and FFT oversampling in the
**Controls** window, press **Compute** (runs the fringe fit on a background thread), then explore
the **UV vs SNR** scatter (click a point to select a fringe) and the selected fringe's **Heatmaps**.
All three windows are dockable on a root dockspace.
"""
interactive(file) = _run(AppState(VLBI.load(VLBI.UVData, file)))

# Set up the ImGui/ImPlot context and run the render loop. `frames` limits the loop to N frames
# (then returns `:imgui_exit_loop`) for headless smoke tests; `nothing` runs until the window closes.
function _run(app::AppState; frames::Union{Int,Nothing}=nothing)
    CImGui.set_backend(:GlfwOpenGL3)
    ctx = CImGui.CreateContext()
    pctx = ImPlot.CreateContext()
    ImPlot.SetImGuiContext(ctx)

    io = CImGui.GetIO()
    io.ConfigFlags = unsafe_load(io.ConfigFlags) | CImGui.ImGuiConfigFlags_DockingEnable

    ImGuiThemes.apply_theme!("Solarized Light")

    n = Ref(0)
    CImGui.render(ctx; window_title="FringeHunt.jl", on_exit=() -> ImPlot.DestroyContext(pctx)) do
        CImGui.DockSpaceOverViewport()
        sync_result!(app)
        ensure_block_cache!(app)
        draw_controls!(app)
        draw_uvsnr!(app)
        draw_heatmaps!(app)
        if frames !== nothing
            n[] += 1
            n[] > frames && return :imgui_exit_loop
        end
        nothing
    end
end


function calculate_timesteps(times::AbstractVector)
	ts = (times .- first(times)) .|> u"s" .|> float
	dt = @p ts diff map(abs) median
	tns = @p ts enumerate() map() do (i, t)
		n, Δ = divrem(t, dt, RoundNearest)
		@assert abs(Δ) < 0.3*dt (dt, t/dt)
		Int(n)
	end
	@assert issorted(tns)
	return (;dt, tns)
end

function fringefit_single(data::KeyedArray; pad_factor::Int)
	@assert dimnames(data) == (:freq, :time)

	fabs = @p let
		data
		fft(zeropad(__; factor=pad_factor))  # actual calculation: pad + fft
		fftshift  # shift zero to be at the center

		# more familiar dimension names and units:
		@set dimnames(__) = (:delay, :rate)
		@modify(d -> d .|> u"ns", __ |> axiskeys(_, :delay))
		@modify(d -> d .|> u"mHz", __ |> axiskeys(_, :rate))

        map(abs)
	end

	# calculate the peak value, and noise distribution:
	med = median(fabs)
	σ = med / median(Rayleigh(1))
	ntrials = length(fabs)
	peakval, peakloc = with_axiskeys(findmax)(fabs)
    value = peakval ±ᵤ σ

	return (; fabs, peakloc, value, σ, ntrials)
end

fringe_pfd(r) = r.ntrials * exp(-0.5 * U.nσ(r.value)^2)


function zeropad(A::AbstractArray; factor)
	P = zeros(eltype(A), size(A) .* factor)
	P[CartesianIndices(A)] .= A
	return P
end
function zeropad(A::KeyedArray; factor)
	KeyedArray(zeropad(AxisKeys.keyless_unname(A); factor); map(ak -> expand_range(ak; factor), named_axiskeys(A))...)
end
expand_range(rng::StepRangeLen; factor::Int) = @set rng.len *= factor
expand_range(rng::LinRange; factor::Int) = LinRange(first(rng), first(rng) + (last(rng) - first(rng)) * factor, length(rng) * factor)



end
