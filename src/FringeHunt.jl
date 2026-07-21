module FringeHunt

using VLBIFiles
import FITSIO
using DataManipulation
using Dictionaries
using Statistics
using CImGui
using CImGui.CSyntax
import GLFW, ModernGL
import ImPlot, ImGuiThemes, ImPlotExtra
using ProgressLogging
using Logging
using Dates: datetime2unix
using StructArrays
using Unitful 
using IntervalSets
using Uncertain
using AxisKeysExtra
using RectiGrids
using Distributions
using LinearAlgebra: norm, mul!
using AccessorsExtra
using FFTW
using OhMyThreads: tmapreduce, TaskLocalValue


function load_source(wide_table, source_id)
    @p wide_table |>
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

# Split a time-sorted record table into maximal contiguous segments that each lie on a single uniform
# time grid. A gap that is not a whole multiple of the sampling step resumes off-grid and can't be placed
# on that grid, so it begins a new segment; gaps that are whole multiples stay within a segment. The
# tolerance separates a real fractional-step offset from timestamp rounding noise.
const _GRID_TOL = 0.1
function split_time_segments(recs)
    length(recs) <= 1 && return [recs]
    ts = @p recs.datetime map((_ - first(recs.datetime)) |> u"s" |> float)   # time from start
    d = diff(ts)
    dt = median(d)                                              # sampling step
    cuts = findall(g -> abs(g/dt - round(g/dt)) > _GRID_TOL, d)
    bounds = [0; cuts; length(recs)]
    [recs[bounds[k]+1 : bounds[k+1]] for k in 1:length(bounds)-1]
end

# All fits for one baseline group, using `ws_dict` (a per-task pool of FFT plans + buffers).
function _process_baseline(bl, uvd, scan_id, fit_crosshands, ws_dict)
    # records on one baseline may span a time-grid discontinuity; fit each single-grid segment on its own
    @p split_time_segments(value(bl)) |>
        filter(seg -> length(seg) >= 2) |>           # need ≥2 samples to fit
        flatmap() do seg
            alldata = records_to_visarray(seg)
            # by default fit only parallel hands (RR/LL); the GUI's "Fit cross-hands" toggle adds RL/LR
            stokeslist = fit_crosshands ? axiskeys(alldata, :stokes) : filter(VLBI.is_parallel_hands, axiskeys(alldata, :stokes))
            tspan = first(seg.datetime) .. last(seg.datetime)
            @p grid(; band=uvd.freq_windows, stokes=stokeslist) vec filtermap() do (;band, stokes)
                freqs = VLBIFiles.frequencies(band)  # per-IF channel frequencies (a StepRangeLen, needed by zeropad)
                data0 = alldata(stokes=stokes)(freq=freqs)
                data = @set named_axiskeys(data0).freq = freqs
                (; value, peakloc, ntrials) = fringefit_peak(data, ws_dict; pad_factor=2)
                (; band, stokes, scan_id, key(bl)..., uv=mean(seg.uvw), tspan,
                   value, peakloc, ntrials)
            end
        end
end

# A scan's baselines in parallel (OhMyThreads); each task owns a workspace pool (`TaskLocalValue`), lock-
# free bar one-time planning. Order-independent (unordered scatter downstream).
function _fit_baselines_parallel(bls, uvd, scan_id, fit_crosshands)
    isempty(bls) && return NamedTuple[]
    pool = TaskLocalValue{Dict{NTuple{2,Int},FFTWorkspace}}(() -> Dict{NTuple{2,Int},FFTWorkspace}())
    tmapreduce(vcat, bls) do bl
        _process_baseline(bl, uvd, scan_id, fit_crosshands, pool[])
    end
end

function compute_fringefits_all(uvd, uvdata_src; fit_crosshands=false)
    # scan-outer: each scan's rows are one contiguous file span, so one readahead replaces per-baseline faults.
    scans = @p uvdata_src group_vg(_.scan_id) collect
    isempty(scans) && return NamedTuple[]
    # overlap I/O with compute: scan i+1's readahead runs on a background task while scan i is fit.
    prefetch(i) = Threads.@spawn VLBIFiles.prefetch!(value(scans[i]).visibility)
    pf = Ref(prefetch(1))
    results = @withprogress name="Fitting fringes" map(enumerate(scans)) do (i, scan)
        wait(pf[])                                        # scan i's visibilities are in RAM
        i < length(scans) && (pf[] = prefetch(i + 1))     # kick off scan i+1's readahead
        bls = @p value(scan) group_vg((; ants=antenna_names(_.baseline))) collect
        res = _fit_baselines_parallel(bls, uvd, key(scan), fit_crosshands)
        @logprogress i / length(scans)
        res
    end
    reduce(vcat, results)
end

function compute_selected_block(uvdata_src, fringe)
    freqs = VLBIFiles.frequencies(fringe.band)
    block = @p uvdata_src |>
        filter(@o _.scan_id == fringe.scan_id) |>            # column-aware: visibility stays lazy
        filter(@o antenna_names(_.baseline) == fringe.ants) |>
        filter(@o _.datetime in fringe.tspan) |>             # the selected fringe's segment
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

# Scatter axis quantities and their native ImPlot scale. The X axis shows `:uvdist` or `:time`
# (user-selectable); the Y axis is always `:snr`.
const AXIS_QUANTITIES = (
    uvdist = (; label="UV distance (km)", scale=:linear),
    time   = (; label="time",             scale=:time),
    snr    = (; label="SNR",              scale=:symlog),
)
const X_OPTIONS = ((:uvdist, "UV distance"), (:time, "time"))
const COLOR_OPTIONS = ((:freq, "frequency"), (:stokes, "stokes"))

_implot_scale(s) = s === :symlog ? ImPlot.ImPlotScale_SymLog :
                   s === :time   ? ImPlot.ImPlotScale_Time   : ImPlot.ImPlotScale_Linear

# Canonical polarization order ⇒ a stable categorical color per stokes (its index into a qualitative
# colormap), so RR/LL/… keep the same color regardless of which subset is present.
const STOKES_ORDER = (:RR, :LL, :RL, :LR, :XX, :YY, :XY, :YX)
const STOKES_COLORMAP = ImPlot.ImPlotColormap_Deep
_stokes_rank(s) = something(findfirst(==(s), STOKES_ORDER), length(STOKES_ORDER) + 1)

# Immutable snapshot of one completed fit, published atomically by the compute task and read by the
# render thread. `points` is one StructArray row per fringe — the fitted fringe fields plus the derived
# scatter columns — giving the render thread both columnar access (`points.snr` straight into
# PlotScatter) and row access (`points[i]` straight into compute_selected_block / the info panel).
struct FitResult
    uvdata_src
    points::StructArray
end

# Midpoint instant of a time interval.
_midtime(span) = leftendpoint(span) + (rightendpoint(span) - leftendpoint(span) |> u"s") / 2

# A frequency (any Hz-convertible Unitful quantity) as text, auto-scaled to kHz/MHz/GHz.
function format_freq(f)
    u = abs(f) ≥ 1u"GHz" ? u"GHz" : abs(f) ≥ 1u"MHz" ? u"MHz" : abs(f) ≥ 1u"kHz" ? u"kHz" : u"Hz"
    string(round(ustrip(u, f); sigdigits=5), " ", u)
end

# Derive the scatter rows from the fitted fringes. `time` is unix seconds (what ImPlot's Time scale
# expects), taken at the segment's midpoint.
function scatter_points(fringefits)
    isempty(fringefits) &&
        return StructArray((; uvdist=Float64[], snr=Float64[], freq=Float64[], time=Float64[]))
    StructArray(map(fringefits) do f
        (; f...,
           uvdist = ustrip(u"km", hypot(f.uv[1], f.uv[2])),   # projected baseline length
           snr    = U.nσ(f.value),
           freq   = ustrip(u"GHz", VLBIFiles.frequency(f.band)),
           time   = datetime2unix(_midtime(f.tspan)))
    end)
end

# All mutable GUI state, owned by the render thread. The compute task only publishes `result` (one
# atomic reference); `running`/`fraction` are atomic. Everything else is render-thread-only.
mutable struct AppState
    const uvd
    wide_table::Any                               # uvtable_wide(uvd), built once by the loader; reused by every Compute
    @atomic source_table::Union{Nothing,StructVector}   # (id, name, nscans, nvis) per source; nothing until loaded
    sel_source::Cint                              # 0-based row index into source_table
    pad_factor::Cint                              # FFT oversampling slider (1..10)
    fit_crosshands::Bool                          # compute setting: also fit RL/LR (applies on next Compute)
    x_quantity::Symbol                            # X axis: :uvdist or :time
    color_quantity::Symbol                        # marker color: :freq or :stokes
    chance_exp::Cint                              # display: chance-probability cutoff line at P = 10^chance_exp (-6..0)

    @atomic result::Union{Nothing,FitResult}      # published by the compute task
    shown::Union{Nothing,FitResult}               # result the render thread has initialized for
    fr_colors::Vector{CImGui.ImU32}               # per-fringe marker color (by the colored quantity)
    color_legend::Vector{Tuple{String,CImGui.ImVec4}}   # discrete stokes legend entries (empty for :freq)
    color_key::Any                                # (result, color_quantity) the marker colors were built for
    selected::Int                                 # row index into shown.points, or 0 if none
    if_filter::Int                                # restrict the scatter to this IF (fw.ix); 0 shows all
    refit_scatter::Bool                           # ask ImPlot to refit the scatter axes next frame

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
    app = AppState(uvd, nothing, nothing, Cint(0), Cint(4), false, :uvdist, :freq, Cint(-3),
                   nothing, nothing, CImGui.ImU32[], Tuple{String,CImGui.ImVec4}[], nothing, 0, 0, false,
                   Threads.Atomic{Bool}(false), FractionLogger(), nothing,
                   nothing, nothing, nothing, nothing, nothing, false)
    # populate the source table on a background thread (one lazy uvtable_wide + per-source scan/vis
    # counts — visibility stays lazy), so the window opens immediately; published atomically when ready.
    Threads.@spawn try
        app.wide_table = uvtable_wide(uvd)            # built once; published before `source_table` gates Compute
        wt = app.wide_table
        source_ids_indata = distinct(wt.source_id)
        srcs = @p let
            sources(uvd)
            getindices(__, source_ids_indata)
            sort(by=_.name)
        end
        @atomic app.source_table = StructArray(map(pairs(srcs)) do (sid, src)
            sub = @p wt filter(@o _.source_id == sid)
            (; id=sid, name=src.name,
               nscans=length(VLBI.scan_intervals(VLBI.GapBasedScans(30u"s"), sub)),
               nvis=length(sub))
        end)
    catch e
        @error "loading source table failed" exception=(e, catch_backtrace())
    end
    app
end

# Kick off the heavy fit on a background thread. The task does ONLY compute and publishes its
# results under `app.lock`; it never touches any ImGui/ImPlot/GL state (that stays on the render
# thread). The render loop polls `app.running` / `app.fl.fraction` each frame for the progress bar.
function start_compute!(app::AppState)
    app.running[] && return
    st = @atomic app.source_table
    isnothing(st) && return                       # sources not loaded yet
    source_id = st.id[app.sel_source + 1]
    wt = app.wide_table                           # cached table, published before `source_table` (read after the gate)
    app.fl.fraction[] = 0.0
    app.running[] = true
    app.task = Threads.@spawn begin
        try
            src, ffs = Logging.with_logger(app.fl) do
                s = load_source(wt, source_id)
                (s, compute_fringefits_all(app.uvd, s; fit_crosshands=app.fit_crosshands))
            end
            # pure DATA only — NO ImPlot/GL calls here. Per-fringe marker colors need the live ImPlot
            # context (ImPlot.SampleColormap), so they are built on the render thread (see ensure_colors!).
            @atomic app.result = FitResult(src, scatter_points(ffs))
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
        app.selected = (isnothing(r) || isempty(r.points)) ? 0 : 1   # initial pick is arbitrary
        app.color_key = nothing                                      # force marker-color rebuild
        app.cache_key = nothing
        app.refit_scatter = true                                     # frame the new data once
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
    fringe = r.points[app.selected]
    block = compute_selected_block(r.uvdata_src, fringe)
    fab = fringefit_single(block; pad_factor=Int(app.pad_factor))
    # stored transposed (image! maps dim1 → x): horizontal = time/rate, vertical = freq/delay
    app.cache_block = permutedims(abs.(block))
    app.cache_phase = permutedims(angle.(block))
    app.cache_fabs = permutedims(fab.fabs)
    app.cache_peak = fab.peakloc
    app.cache_key = key
    app.refit_axes = true
    nothing
end

# Build the per-fringe marker colors for the selected color quantity: continuous viridis over the
# `freq` column, or a discrete categorical color per `stokes` (also filling `color_legend`). MUST run
# on the render thread: ImPlot.SampleColormap/GetColormapColor need the live ImPlot context (segfault
# otherwise). Rebuilt only when the result or the color quantity changes.
function ensure_colors!(app::AppState)
    r = app.shown
    key = (r, app.color_quantity)
    app.color_key == key && return
    app.color_key = key
    app.color_legend = Tuple{String,CImGui.ImVec4}[]
    if isnothing(r) || isempty(r.points)
        app.fr_colors = CImGui.ImU32[]
        return
    end
    if app.color_quantity === :stokes
        uniq = sort(unique(r.points.stokes); by=_stokes_rank)
        v4 = Dict(s => ImPlot.GetColormapColor(i - 1, STOKES_COLORMAP) for (i, s) in enumerate(uniq))
        u32 = Dict(s => CImGui.ColorConvertFloat4ToU32(v4[s]) for s in uniq)
        app.fr_colors = map(s -> u32[s], r.points.stokes)
        app.color_legend = [(string(s), v4[s]) for s in uniq]
    else  # :freq — continuous viridis normalized to the column extrema
        col = r.points.freq
        lo, hi = extrema(col)
        app.fr_colors = map(col) do v
            t = hi > lo ? (v - lo) / (hi - lo) : 0.5
            CImGui.ColorConvertFloat4ToU32(ImPlot.SampleColormap(Cfloat(t), ImPlot.ImPlotColormap_Viridis))
        end
    end
    nothing
end

# Closed interval spanning the first..last axis key (unit-stripped), for image! extents.
_axis_interval(ax, unit) = (v = ustrip.(unit, ax); minimum(v) .. maximum(v))

# Pixel-edge bounds of an `n`-cell image spanning `int` (cell centers) — matches what `image!` draws.
_image_bounds(int, n) = (a = leftendpoint(int); b = rightendpoint(int);
                         Δ = n > 1 ? (b - a) / (n - 1) : (b > a ? b - a : one(b)); (a - Δ/2, b + Δ/2))

# Clamp a heatmap's pan/zoom to the image extent. The constraint must be the *image* bounds (½ cell
# beyond the key range), not the key range itself: `SetNextAxesToFit` fits to the image quad, and a
# fit target even slightly outside the constraint is rejected wholesale (so the fit silently no-ops).
function _heatmap_constraints(xint, yint, data)
    bx = _image_bounds(xint, size(data, 1))
    by = _image_bounds(yint, size(data, 2))
    ImPlot.SetupAxisLimitsConstraints(ImPlot.ImAxis_X1, bx...)
    ImPlot.SetupAxisLimitsConstraints(ImPlot.ImAxis_Y1, by...)
end

# One selectable row per IF (number, channel-center range, channel spacing), with a dim static gap row
# between consecutive IFs (nearest-edge separation, negative = overlap). Selecting an IF restricts the
# scatter to its points; clicking the selected IF again clears the filter.
function draw_if_table!(app::AppState)
    fws = app.uvd.freq_windows
    CImGui.BeginTable("##ifs", 3) || return
    CImGui.TableSetupColumn("IF")
    CImGui.TableSetupColumn("range")
    CImGui.TableSetupColumn("Δ chan")
    CImGui.TableHeadersRow()
    for (k, fw) in enumerate(fws)
        lo, hi = extrema(VLBIFiles.frequencies(fw))
        CImGui.TableNextRow()
        CImGui.TableNextColumn()
        sel = app.if_filter == fw.ix
        CImGui.Selectable("$(fw.ix)##if", sel, CImGui.ImGuiSelectableFlags_SpanAllColumns) &&
            (app.if_filter = sel ? 0 : fw.ix)                # toggle off when re-clicking the selected IF
        CImGui.TableNextColumn(); CImGui.Text("$(format_freq(lo)) – $(format_freq(hi))")
        CImGui.TableNextColumn(); CImGui.Text(format_freq(fw.width / fw.nchan))
        if k < length(fws)
            lb, hb = extrema(VLBIFiles.frequencies(fws[k+1]))
            gap = max(lo, lb) - min(hi, hb)                  # nearest-edge separation, sideband-agnostic
            CImGui.TableNextRow()
            CImGui.TableNextColumn(); CImGui.TableNextColumn()
            CImGui.TextColored(CImGui.ImVec4(0.5, 0.5, 0.5, 1), "gap $(format_freq(gap))")
        end
    end
    CImGui.EndTable()
end

function draw_controls!(app::AppState)
    CImGui.Begin("Controls")
    st = @atomic app.source_table
    if isnothing(st)
        CImGui.Text("Loading sources...")
    elseif CImGui.BeginTable("##sources", 2)
        for (i, s) in enumerate(st)
            CImGui.TableNextRow()
            CImGui.TableNextColumn()
            # the source name IS the radio-button label (clicking it selects); `##id` keeps the id unique
            CImGui.RadioButton("$(s.name)##$(s.id)", app.sel_source == i - 1) && (app.sel_source = Cint(i - 1))
            CImGui.TableNextColumn()
            CImGui.Text("$(s.nscans) scans, $(s.nvis) vis")
        end
        CImGui.EndTable()
    end

    CImGui.SeparatorText("Frequency bands (IFs)")
    draw_if_table!(app)
    CImGui.Separator()

    pf = Ref(app.pad_factor)
    CImGui.SliderInt("", pf, Cint(1), Cint(10), "FFT oversampling: %d×")
    app.pad_factor = pf[]

    ch = Ref(app.fit_crosshands)                  # compute setting: applies on next Compute
    CImGui.Checkbox("Fit cross-hands (RL/LR)", ch)
    app.fit_crosshands = ch[]

    running = app.running[]
    busy = running || isnothing(st)               # can't compute until sources are loaded
    busy && CImGui.BeginDisabled()
    CImGui.Button("Compute") && start_compute!(app)
    busy && CImGui.EndDisabled()

    if running
        CImGui.SameLine()
        CImGui.ProgressBar(Cfloat(app.fl.fraction[]), CImGui.ImVec2(-1, 0), "")
    elseif (r = app.shown) !== nothing
        CImGui.Text("$(length(r.points)) fringes fitted")
        if app.selected != 0
            f = r.points[app.selected]
            CImGui.Text("Selected: $(f.ants[1])-$(f.ants[2]) $(f.stokes) scan $(f.scan_id)")
            CImGui.Text(string("SNR = ", round(f.snr; digits=1),
                               ", UV = ", round(f.uvdist; digits=0), " km"))
        end
    end
    CImGui.End()
end

# Clamp an axis's pan/zoom to its data range with 5% padding (degenerate range guarded).
function _scatter_axis_constraint(axis, vals)
    lo, hi = extrema(vals)
    pad = hi > lo ? 0.05 * (hi - lo) : (iszero(hi) ? 1.0 : 0.05 * abs(hi))
    ImPlot.SetupAxisLimitsConstraints(axis, lo - pad, hi + pad)
end

# A one-line radio group: `label` then a radio per option. Returns the (possibly changed) selection.
function radio_group(label, current::Symbol, options)
    CImGui.Text(label)
    sel = current
    for (val, txt) in options
        CImGui.SameLine()
        CImGui.RadioButton("$txt##$label", current == val) && (sel = val)
    end
    sel
end

const _SWATCH_FLAGS = CImGui.ImGuiColorEditFlags_NoTooltip | CImGui.ImGuiColorEditFlags_NoDragDrop

function draw_uvsnr!(app::AppState)
    CImGui.Begin("UV vs SNR")
    r = app.shown
    if isnothing(r)
        CImGui.Text("Press Compute to fit fringes.")
        CImGui.End()
        return
    end

    # per-axis selectors on one line: X is uvdist/time, color is frequency/stokes (Y is always SNR)
    newx = radio_group("X axis:", app.x_quantity, X_OPTIONS)
    newx == app.x_quantity || (app.x_quantity = newx; app.refit_scatter = true)   # refit once on axis change
    CImGui.SameLine(0, 30)
    app.color_quantity = radio_group("color:", app.color_quantity, COLOR_OPTIONS)
    CImGui.SameLine(0, 30)
    ce = Ref(app.chance_exp)                       # chance-probability cutoff line on the scatter
    CImGui.SetNextItemWidth(200)
    CImGui.SliderInt("##chance", ce, Cint(-6), Cint(0), "chance p: 1e%d")
    app.chance_exp = ce[]

    ensure_colors!(app)

    xq = app.x_quantity
    # show all points, or just the selected IF's — subset the one points table; `idxs` maps back to r.points rows
    if app.if_filter == 0
        pts, colors, idxs = r.points, app.fr_colors, eachindex(r.points)
    else
        idxs = findall(b -> b.ix == app.if_filter, r.points.band)
        pts = r.points[idxs]
        colors = isempty(app.fr_colors) ? app.fr_colors : app.fr_colors[idxs]
    end
    xs, ys = getproperty(pts, xq), pts.snr
    sp = unsafe_load(CImGui.GetStyle()).ItemSpacing.x
    cbw = 90.0f0                                   # reserve right-pane width for the colorbar / legend
    avail = CImGui.GetContentRegionAvail()
    plot_w = avail.x - cbw - sp

    refit = app.refit_scatter
    app.refit_scatter = false
    refit && ImPlot.SetNextAxesToFit()

    if ImPlot.BeginPlot("##uvsnr", AXIS_QUANTITIES[xq].label, AXIS_QUANTITIES[:snr].label,
                        CImGui.ImVec2(plot_w, -1); flags=ImPlot.ImPlotFlags_NoLegend)
        ImPlot.SetupAxisScale(ImPlot.ImAxis_X1, _implot_scale(AXIS_QUANTITIES[xq].scale))
        ImPlot.SetupAxisScale(ImPlot.ImAxis_Y1, _implot_scale(AXIS_QUANTITIES[:snr].scale))
        if !isempty(xs)
            _scatter_axis_constraint(ImPlot.ImAxis_X1, xs)
            _scatter_axis_constraint(ImPlot.ImAxis_Y1, ys)
        end
        if isempty(colors)
            ImPlot.PlotScatter("fringes", xs, ys; spec=ImPlot.ImPlotSpec(Marker=ImPlot.ImPlotMarker_Circle))
        else
            GC.@preserve colors begin
                # color on the marker EDGE, transparent fill
                spec = ImPlot.ImPlotSpec(Marker=ImPlot.ImPlotMarker_Circle,
                    MarkerLineColors=pointer(colors), MarkerFillColor=CImGui.ImVec4(0, 0, 0, 0))
                ImPlot.PlotScatter("fringes", xs, ys; spec)
            end
        end
        # highlight the selected fringe with a larger distinct marker
        if app.selected != 0
            f = r.points[app.selected]
            ImPlot.PlotScatter("selected", [getproperty(f, xq)], [f.snr];
                spec=ImPlot.ImPlotSpec(Marker=ImPlot.ImPlotMarker_Circle, MarkerSize=9,
                    MarkerFillColor=CImGui.ImVec4(1, 0, 0, 0.0), MarkerLineColor=CImGui.ImVec4(1, 0, 0, 1)))
        end
        # click-to-select: nearest fringe in PIXEL space (axes may be SymLog/Time ⇒ data-space distance is meaningless)
        if ImPlot.IsPlotHovered() && CImGui.IsMouseClicked(0) && !isempty(xs)
            mp = ImPlot.GetPlotMousePos()
            mpx = ImPlot.PlotToPixels(mp.x, mp.y)
            best = argmin(eachindex(xs)) do i
                px = ImPlot.PlotToPixels(xs[i], ys[i])
                (px.x - mpx.x)^2 + (px.y - mpx.y)^2
            end
            app.selected = idxs[best]
        end
        # chance-probability cutoff: SNR s where N·exp(-½s²) = P (N = # cells), as a horizontal line + axis tag
        if !isempty(ys)
            N = median(r.points.ntrials)
            scut = cquantile(Rayleigh(1), 10.0^app.chance_exp / N)
            ImPlot.PlotInfLines("##chance", [scut];
                spec=ImPlot.ImPlotSpec(Flags=ImPlot.ImPlotInfLinesFlags_Horizontal, LineColor=CImGui.ImVec4(1, 0, 0, 1)))
            ImPlot.TagY(scut, CImGui.ImVec4(1, 0, 0, 1), string(round(scut; digits=1)))
        end
        ImPlot.EndPlot()
        # right pane: continuous colorbar (frequency) or a discrete swatch legend (stokes)
        CImGui.SameLine()
        if app.color_quantity === :stokes
            CImGui.BeginGroup()
            for (lbl, c) in app.color_legend
                CImGui.ColorButton("##sw_$lbl", c, _SWATCH_FLAGS, CImGui.ImVec2(14, 14))
                CImGui.SameLine()
                CImGui.Text(lbl)
            end
            CImGui.EndGroup()
        else
            lo, hi = isempty(r.points) ? (0.0, 1.0) : extrema(r.points.freq)
            ImPlot.ColormapScale("frequency (GHz)", lo, hi,
                CImGui.ImVec2(cbw, -1), "%g", 0, ImPlot.ImPlotColormap_Viridis)
        end
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

    fabs = app.cache_fabs                                        # (rate, delay): rate horizontal, delay vertical
    rate_int  = _axis_interval(axiskeys(fabs, :rate), u"mHz")
    delay_int = _axis_interval(axiskeys(fabs, :delay), u"ns")
    peak_rate  = ustrip(u"mHz", app.cache_peak.rate)
    peak_delay = ustrip(u"ns", app.cache_peak.delay)

    avail = CImGui.GetContentRegionAvail()
    top_h = avail.y * 0.55f0

    refit && ImPlot.SetNextAxesToFit()   # fit once on change; otherwise the user can zoom/pan freely within bounds
    if ImPlot.BeginPlot("Fringe Amplitude", "rate (mHz)", "delay (ns)", CImGui.ImVec2(-1, top_h);
                        flags=ImPlot.ImPlotFlags_NoLegend)
        _heatmap_constraints(rate_int, delay_int, fabs)
        ImPlotExtra.image!("fringe", rate_int, delay_int, fabs; colormap=:viridis, colorscale=log10, interpolate=false)
        ImPlot.PlotInfLines("##peakrate", [peak_rate])
        ImPlot.PlotInfLines("##peakdelay", [peak_delay]; spec=ImPlot.ImPlotSpec(Flags=ImPlot.ImPlotInfLinesFlags_Horizontal))
        ImPlot.EndPlot()
    end

    block = app.cache_block                                      # (time, freq): time horizontal, freq vertical
    phase = app.cache_phase
    time_int = _axis_interval(axiskeys(block, :time), u"s")
    freq_int = _axis_interval(axiskeys(block, :freq), u"GHz")
    half_w = CImGui.GetContentRegionAvail().x * 0.5f0 - 4

    refit && ImPlot.SetNextAxesToFit()
    if ImPlot.BeginPlot("Data Amplitude", "time (s)", "freq (GHz)", CImGui.ImVec2(half_w, -1);
                        flags=ImPlot.ImPlotFlags_NoLegend)
        _heatmap_constraints(time_int, freq_int, block)
        ImPlotExtra.image!("amp", time_int, freq_int, block; colormap=:viridis, interpolate=false)
        ImPlot.EndPlot()
    end
    CImGui.SameLine()
    refit && ImPlot.SetNextAxesToFit()
    if ImPlot.BeginPlot("Data Phase", "time (s)", "freq (GHz)", CImGui.ImVec2(-1, -1);
                        flags=ImPlot.ImPlotFlags_NoLegend)
        _heatmap_constraints(time_int, freq_int, phase)
        ImPlotExtra.image!("phase", time_int, freq_int, phase; colormap=:twilight,
                           colorrange=(-Float32(pi), Float32(pi)), interpolate=false)
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

# ---------------------------------------------------------------------------
# Fast batched fringe-fit core (used by compute_fringefits_all)
# ---------------------------------------------------------------------------
# Peak-only path, called thousands of times: reuses buffers, skips fftshift/KeyedArray. peakloc/ntrials
# match `fringefit_single`; SNR differs only by the subsampled noise median.

const _PLAN_LOCK = ReentrantLock()            # FFTW planning is not thread-safe; guard workspace creation

# Noise median from a stride-7 subsample of |FFT|² (exact median was the top cost; uniform floor → ≈1%).
# 5-smooth sizes are never ÷7, so the stride stays coprime with every row count and covers all delay rows.
const _MEDIAN_STRIDE = 7

# Reusable per-(padded size) scratch: an FFT plan plus the buffers a single fit touches.
struct FFTWorkspace{P}
    plan::P                      # out-of-place complex FFT plan for `inbuf`
    inbuf::Matrix{ComplexF32}    # zeropad target (padded size)
    outbuf::Matrix{ComplexF32}   # FFT output (mul! destination, unshifted)
    submag::Vector{Float32}      # strided subsample of |outbuf|², for the approximate noise median
end

# 5-smooth padded length ≥ n*factor: FFTW is slow on large prime factors; rounding up only refines the grid.
_padded_len(n, factor) = nextprod((2, 3, 5), n * factor)

function _get_workspace!(ws_dict, padsize)
    get!(ws_dict, padsize) do
        lock(_PLAN_LOCK) do
            inbuf = zeros(ComplexF32, padsize)
            # MEASURE (ESTIMATE's plan is ~4× slower here); ~25µs, amortized over thousands of fits. Overwrites inbuf.
            plan = plan_fft(inbuf; flags=FFTW.MEASURE)
            submag = Vector{Float32}(undef, cld(prod(padsize), _MEDIAN_STRIDE))
            FFTWorkspace(plan, inbuf, similar(inbuf), submag)
        end
    end
end

# Fast `findmax(f, out)`: Base's won't SIMD (index-carrying, NaN-aware reduction), ~3× slower. 4 lanes break
# the loop-carried dependency. Pass `abs2` (skips a sqrt). NB: does not propagate NaN.
function _findmax(f, out)
    n = length(out)
    @inbounds a1 = a2 = a3 = a4 = f(out[1])
    i1 = i2 = i3 = i4 = 1
    k = 1
    @inbounds while k + 3 <= n
        m1 = f(out[k]);     m1 > a1 && (a1 = m1; i1 = k)
        m2 = f(out[k + 1]); m2 > a2 && (a2 = m2; i2 = k + 1)
        m3 = f(out[k + 2]); m3 > a3 && (a3 = m3; i3 = k + 2)
        m4 = f(out[k + 3]); m4 > a4 && (a4 = m4; i4 = k + 3)
        k += 4
    end
    @inbounds while k <= n; m = f(out[k]); m > a1 && (a1 = m; i1 = k); k += 1 end
    a2 > a1 && (a1 = a2; i1 = i2); a3 > a1 && (a1 = a3; i1 = i3); a4 > a1 && (a1 = a4; i1 = i4)
    (a1, CartesianIndices(out)[i1])   # loop stays linear for speed; convert once, matching Base findmax
end

# One fit: zeropad → FFT → peak + subsampled-median noise. Works in |·|² (√ only on the two outputs).
function _fringefit_core(data::KeyedArray, ws::FFTWorkspace)
    @assert dimnames(data) == (:freq, :time)
    M = AxisKeys.keyless_unname(data)                # raw ComplexF32 matrix (nf, nt)
    nf, nt = size(M)
    fill!(ws.inbuf, 0)
    @inbounds @views ws.inbuf[1:nf, 1:nt] .= M       # zeropad into the reused buffer
    mul!(ws.outbuf, ws.plan, ws.inbuf)               # unshifted FFT
    out = ws.outbuf
    peakabs2, peakidx = _findmax(abs2, out)
    peakval = sqrt(peakabs2)
    ws.submag .= abs2.(@view vec(out)[1:_MEDIAN_STRIDE:end])   # strided subsample for the noise median
    σ = sqrt(median!(ws.submag)) / median(Rayleigh(1))         # median(|z|) / median(Rayleigh(1)); folds to a const
    Nf, Nt = size(out)
    i, j = Tuple(peakidx)                            # unshifted argmax → (delay idx, rate idx)
    Δf = step(AxisKeys.axiskeys(data, :freq))
    Δt = step(AxisKeys.axiskeys(data, :time))
    delay = FFTW.fftfreq(Nf, 1 / Δf)[i] |> u"ns"     # AxisKeys' `fft` uses this same FFTW.fftfreq, so they agree
    rate  = FFTW.fftfreq(Nt, 1 / Δt)[j] |> u"mHz"
    (; value = peakval ±ᵤ σ, peakloc = (; delay, rate), ntrials = length(out))
end

# Workspace-pooled peak-only fit used by the batch driver.
function fringefit_peak(data::KeyedArray, ws_dict; pad_factor::Int)
    nf, nt = size(data)
    padsize = (_padded_len(nf, pad_factor), _padded_len(nt, pad_factor))
    _fringefit_core(data, _get_workspace!(ws_dict, padsize))
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
	P = zeros(eltype(A), map(n -> _padded_len(n, factor), size(A)))
	P[CartesianIndices(A)] .= A
	return P
end
function zeropad(A::KeyedArray; factor)
	KeyedArray(zeropad(AxisKeys.keyless_unname(A); factor); map(ak -> expand_range(ak; factor), named_axiskeys(A))...)
end
# expand a key range to the padded length, keeping its step (so the delay/rate grid matches the batch core)
expand_range(rng::StepRangeLen; factor::Int) = @set rng.len = _padded_len(rng.len, factor)
expand_range(rng::LinRange; factor::Int) =
	(N = _padded_len(length(rng), factor); LinRange(first(rng), first(rng) + (last(rng) - first(rng)) * (N - 1) / (length(rng) - 1), N))



end
