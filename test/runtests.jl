using TestItems
using TestItemRunner
@run_package_tests


@testitem "_" begin
    import Aqua
    # skip persistent_tasks: it resolves the package in an isolated env, which can't see the
    # unregistered git/dev deps (ImPlot fork, ImPlotExtra) and errors with "no known versions"
    Aqua.test_all(FringeHunt; persistent_tasks=false)

    import CompatHelperLocal as CHL
    CHL.@check()
end

@testitem "FractionLogger" begin
    # ProgressLogging/Logging accessed via FringeHunt (its deps) to avoid extra test deps
    fl = FringeHunt.FractionLogger()
    fracs = Float64[]
    FringeHunt.Logging.with_logger(fl) do
        FringeHunt.ProgressLogging.@withprogress begin
            for i in 1:10
                FringeHunt.ProgressLogging.@logprogress i / 10
                push!(fracs, fl.fraction[])
            end
        end
    end
    @test fracs ≈ (1:10) ./ 10
    @test fl.fraction[] ≈ 1.0
end

@testitem "split_time_segments" begin
    # types accessed via FringeHunt (its deps) to avoid extra test deps; times are fed as seconds,
    # since split_time_segments only uses their differences
    recs(secs) = FringeHunt.StructArrays.StructArray((; datetime = secs .* FringeHunt.Unitful.u"s"))
    split = FringeHunt.split_time_segments

    # a contiguous run on a uniform grid stays one segment
    @test length(split(recs(0:4:40))) == 1
    # a gap that is a whole multiple of the step (a dropped sample) stays one segment
    @test length(split(recs([0, 4, 8, 20, 24, 28]))) == 1     # 8→20 is 12s = 3×4s
    # a gap offset by a fraction of the step splits into two grids
    parts = split(recs([0, 4, 8, 14, 18, 22]))                # 8→14 is 6s = 1.5×4s
    @test length(parts) == 2
    @test map(length, parts) == [3, 3]
    # a single record is its own segment
    @test length(split(recs([0]))) == 1
end

@testitem "fringefit_peak matches fringefit_single" begin
    # The fast batch core (fringefit_peak) must agree with the reference fringefit_single: identical
    # peak cell (delay/rate) and ntrials — those are independent of the noise median — and SNR within
    # the subsampled-median tolerance. Types via FringeHunt's deps; Random is stdlib.
    using FringeHunt.AxisKeys, FringeHunt.Unitful
    import Random
    Random.seed!(42)
    for (nf, nt) in ((32, 64), (40, 50), (128, 100))
        freqs = range(8.0u"GHz", step=2.0f0u"MHz", length=nf)
        times = range(0.0u"s", step=1.0u"s", length=nt)
        # a synthetic fringe (delay-like phase ramp over freq, rate-like over time) + noise → a single
        # clear peak, so the two paths can't disagree by tie-breaking
        fringe = [cis(2π * (ustrip(u"GHz", f) * 50.0 + ustrip(u"s", t) * 0.005)) for f in freqs, t in times]
        d = KeyedArray(ComplexF32.(fringe) .+ 0.3f0 .* randn(ComplexF32, nf, nt); freq=freqs, time=times)

        ref = FringeHunt.fringefit_single(d; pad_factor=2)
        new = FringeHunt.fringefit_peak(d, Dict{NTuple{2,Int}, FringeHunt.FFTWorkspace}(); pad_factor=2)

        @test new.ntrials == ref.ntrials
        # same peak cell (the two paths derive the delay/rate value with last-bit-different rounding)
        @test new.peakloc.delay ≈ ref.peakloc.delay rtol = 1e-4
        @test new.peakloc.rate ≈ ref.peakloc.rate rtol = 1e-4
        @test FringeHunt.U.nσ(new.value) ≈ FringeHunt.U.nσ(ref.value) rtol = 0.1
    end
end

@testitem "smooth padding refines, never coarsens" begin
    # _padded_len oversamples by at least `factor`, rounded up to a 5-smooth (FFT-friendly) size
    pad = FringeHunt._padded_len
    for n in (50, 100, 128, 414, 459, 511), factor in (1, 2, 4)
        L = pad(n, factor)
        @test L >= n * factor                       # never less oversampling than requested
        @test FringeHunt.nextprod((2, 3, 5), L) == L  # 5-smooth
    end
    @test pad(459, 2) == 960                          # the motivating case: 918 (=2·3³·17) → 960
end

@testitem "IF grouping: band_groups, combined_grid, band_contiguous" begin
    using FringeHunt.VLBIFiles, FringeHunt.Unitful
    FW = VLBIFiles.FrequencyWindow
    a = FW(1, 1, 8.000f9u"Hz", 8f6u"Hz", 4, 1, 1f0)   # 4 channels, 2 MHz step
    b = FW(1, 2, 8.008f9u"Hz", 8f6u"Hz", 4, 1, 1f0)   # contiguous with a
    c = FW(1, 3, 8.030f9u"Hz", 8f6u"Hz", 4, 1, 1f0)   # gap after b
    fws = [a, b, c]

    @test FringeHunt.band_groups(FringeHunt.PerIF(), fws) == [[a], [b], [c]]
    @test FringeHunt.band_groups(FringeHunt.ManualCombine([2, 1]), fws) == [[a, b]]   # frequency-sorted

    g = FringeHunt.combined_grid([a, b])
    @test g isa StepRangeLen
    @test g ≈ vcat(VLBIFiles.frequencies(a), VLBIFiles.frequencies(b))               # exact concatenation

    @test FringeHunt.band_contiguous([a, b])
    @test !FringeHunt.band_contiguous([a, c])
    @test FringeHunt.band_contiguous([a])
end

@testitem "combined-grid fit recovers the injected delay/rate" begin
    using FringeHunt.VLBIFiles, FringeHunt.AxisKeys, FringeHunt.Unitful
    FW = VLBIFiles.FrequencyWindow
    # two contiguous IFs → one 16-channel, 32 MHz band
    a = FW(1, 1, 8.000f9u"Hz", 16f6u"Hz", 8, 1, 1f0)
    b = FW(1, 2, 8.016f9u"Hz", 16f6u"Hz", 8, 1, 1f0)
    freqs = FringeHunt.combined_grid([a, b])
    times = range(0.0u"s", step=1.0u"s", length=120)
    delay, rate = 40.0u"ns", 3.0u"mHz"
    fringe = [cis(2π * NoUnits(f * delay + t * rate)) for f in freqs, t in times]
    d = KeyedArray(ComplexF32.(fringe); freq=freqs, time=times)

    fab = FringeHunt.fringefit_single(d; pad_factor=8)
    dax, rax = axiskeys(fab.fabs, :delay), axiskeys(fab.fabs, :rate)
    @test fab.peakloc.delay ≈ delay atol=abs(dax[2] - dax[1])   # within one grid cell
    @test fab.peakloc.rate  ≈ rate  atol=abs(rax[2] - rax[1])
end
