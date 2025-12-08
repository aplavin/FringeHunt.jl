module FringeHunt

using VLBIFiles
import FITSIO
using DataManipulation
using Statistics
using MakieExtra; import GLMakie
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


function interactive(file)
    uvd = VLBI.load(VLBI.UVData, file)
    sourcetbl = @p FITSIO.FITS(file)["SOURCE"] StructArrays.fromtable

    update_theme!(inspectable=false)
    fig = Figure(size=(1000, 900))

    srcmenu = fig[1,1:1][1,1] = Menu(fig, options=@p sourcetbl sort(by=_.SOURCE) map((_.SOURCE, @oget _.SOURCE_ID _.var"ID_NO."))) # vcat([("---", -1)], __))
    source_id = srcmenu.selection

    uvdata_src_raw = @lift @p VLBIFiles.read_data_raw(uvd) filter($source_id == -1 ? Returns(false) : @o _.SOURCE == $source_id)
    uvdata_src = @lift @p let
		$uvdata_src_raw
		mapinsert⁻(datetime=@o VLBIFiles.DateTime_from_DATE_TIME(_.DATE, _.TIME))
		mapinsert(spec=@o VisSpec(
			VLBIFiles.Baseline_from_fits(_.BASELINE, uvd.ant_arrays),
			UV_from_uvrow(_)))
		mapset(FLUX=r -> @p let
			r.FLUX
			__[RA=1, DEC=1]  # always trivial dimensions?
			complex.(__(COMPLEX=:re), __(COMPLEX=:im))
			@set __ |> axiskeys(_, :BAND) = @p uvd.freq_windows filter(_.freqid == r.FREQID)
		end)
		delete(__, @o _.var"UU---SIN" _.var"VV---SIN" _.var"WW---SIN" _.var"UU--SIN" _.var"VV--SIN" _.var"WW--SIN" _.var"UU-L" _.var"VV-L" _.var"WW-L" _.BASELINE _.FREQID)
		filter(!allequal(antenna_names(_)))
		VLBI.add_scan_ids(VLBI.GapBasedScans(30u"s"))
	end

    fringefits_all = @lift @p $uvdata_src |>
        group_vg((;named_axiskeys(_.FLUX).BAND, _.scan_id, ants=antenna_names(_))) |>
        collect |>
        flatmap() do scan
            alldata = uvtable_to_visarray(value(scan))
            bandsets = axiskeys(alldata, :BAND)
            stokes = @p axiskeys(alldata, :STOKES) filter(VLBIData.is_parallel_hands(_))
            @p grid(; band=bandsets, stokes) vec filtermap() do (;band, stokes)
                freqs = band isa VLBI.FrequencyWindow ? VLBIFiles.frequencies(band) : @p band flatmap(VLBIFiles.frequencies) vec_to_range()
                data = @p alldata(;BAND=band, STOKES=stokes) |>
                    reshape(__, (:, length(axiskeys(__, :time)))) |>
                    KeyedArray(__, FREQ=freqs, time=axiskeys(alldata, :time))

                (;fabs, value, peakloc, ntrials) = fringefit_single(data; pad_factor=2)
            
                (; band, stokes, key(scan)..., uv=mean(UV, scan), value, peakloc, ntrials)
            end
        end

    # fringe_color = AxFunc(label="Probability of false detection", scale=log10, limit=1e-7..1, @o fringe_pfd(_)),
    fringe_color = AxFunc(label="Frequency (GHz)", scale=identity, @o ustrip(u"GHz", VLBIFiles.frequency(_.band)))

    selected_fringe = Observable(fringefits_all[][1])
	fplt = @lift FPlot(
		$fringefits_all,
		(@o norm(_.uv) |> ustrip(u"km", _)),
		AxFunc(scale=SymLog(5), limit=(2, nothing), ticks=BaseMulTicks([1,2,5]), label="SNR", @o U.nσ(_.value));
		color=$fringe_color,
        marker=(@o _ == $selected_fringe ? '⚫' : '∘'),
        markersize=(@o _ == $selected_fringe ? 30 : 20),
        inspectable=true, inspector_hover=x -> (selected_fringe[] = x; true),
	)
	ax,plt = axplot(scatter)(fig[2,1][1,1], fplt)
	Colorbar(fig[2,1][1,2], plt; height=180, tellheight=false, label=lift(x->x.label, fringe_color))
	
    selected_data_block = @lift @p let
		uvdata_src[]
		filter(_.scan_id == $selected_fringe.scan_id && antenna_names(_) == $selected_fringe.ants)
		uvtable_to_visarray(__)
		__(STOKES=$selected_fringe.stokes, BAND=$selected_fringe.band)
        @set __ |> named_axiskeys(_) = (
            FREQ=VLBIFiles.frequencies($selected_fringe.band),
            time=axiskeys(__, :time)
        )
	end

	heatmap(fig[3,1], (@lift abs.($selected_data_block)), axis=(;title="Data Amplitude"))
	heatmap(fig[3,2], (@lift angle.($selected_data_block)), colormap=:cyclic_mrybm_35_75_c68_n256, axis=(;title="Data Phase"))

    pad_factor, = Slider₊(GridLayout(fig[1,1:2][1,2], tellwidth=false)[1,1:2], range=1:10, startvalue=4, label="FFT oversampling", width=150)
    selected_fringed_blk = @lift fringefit_single($selected_data_block; pad_factor=$pad_factor)

	heatmap(fig[2,2], (@lift $selected_fringed_blk.fabs), colorscale=(@lift SymLog(5median($selected_fringed_blk.fabs))), axis=(;title="Fringe Amplitude"))
	vlines!((@lift $selected_fringed_blk.peakloc.delay |> ustrip); color=:black, linestyle=:dash, linewidth=1)
	hlines!((@lift $selected_fringed_blk.peakloc.rate |> ustrip); color=:black, linestyle=:dash, linewidth=1)

	on(selected_fringed_blk) do _
		# different IFs – different frequency ranges
		for a in contents(fig[3,1:2])
			reset_limits!(a)
		end
	end

    DataInspector(fig)
    GLMakie.activate!(focus_on_show=true, title="FringeHunt.jl")
    display(fig)
    MakieExtra.show_gl_icon_in_dock()
end


UV_from_uvrow(r) =
	UV(
		(@oget r.var"UU---SIN" r.var"UU--SIN" r.var"UU-L"),
		(@oget r.var"VV---SIN" r.var"VV--SIN" r.var"VV-L")
	) .* u"c*s" .|> u"m"

function uvtable_to_visarray(uvtable::StructVector{<:NamedTuple})
	(;dt, tns) = calculate_timesteps(uvtable.datetime)
	ns = @p tns extrema() range(__...)
	Z = zero(uvtable[1].FLUX)
	alldata = @p let
		ns
		map() do n
			ix = searchsorted(tns, n)
			isempty(ix) && return Z
			return uvtable.FLUX[only(ix)]
		end
		stack(KeyedArray(__, time=ns .* dt))
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
	@assert dimnames(data) == (:FREQ, :time)

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
