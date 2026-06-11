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
