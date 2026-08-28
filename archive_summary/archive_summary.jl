#!/usr/bin/env julia

using ArgParse
using DateFormats
using VirtualObservatory

const TAP_SERVICE = TAPService("https://data-query.nrao.edu/tap")

function archive_scans(file)
    product = replace(basename(file), "'" => "''")
    query = """
        SELECT scan_num, target_name, t_min, t_max, t_exptime
        FROM tap_schema.obscore
        WHERE obs_publisher_did='$product'
        ORDER BY scan_num
        """
    rows = execute(TAP_SERVICE, query)
    isempty(rows) && error("no archive scans found for $(basename(file))")
    rows
end

function report(file; io=stdout)
    println(io, basename(file))
    println(io, "scan  source     start – stop  duration")
    foreach(archive_scans(file)) do row
        println(io, rpad(string(row.scan_num), 6), rpad(row.target_name, 11),
                mjd(row.t_min), " – ", mjd(row.t_max), "  ", row.t_exptime, " s")
    end
end

function main(args)
    settings = ArgParseSettings()
    @add_arg_table! settings begin
        "--file"
            help = "exact NRAO archive product name"
            required = true
    end
    report(parse_args(args, settings; as_symbols=true)[:file])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
