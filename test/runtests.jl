using TestItems
using TestItemRunner
@run_package_tests


@testitem "_" begin
    import Aqua
    Aqua.test_all(FringeHunt)

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
