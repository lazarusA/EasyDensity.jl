module Helpers

using JSON
using ArchGDAL
using Proj
using DataFrames
using Rasters
using Base.Threads
const AG = ArchGDAL

export convert_bbox_wgs84_to_3035, make_grid_3035, sample_tiff_onto_grid, write_geotiff_from_grid, load_last_epoch, SOCD_model, make_tiles, preprocess_predictors!

 # because it's always 4326, so we do it in lazy way
const TF_3035_TO_4326 = Proj.Transformation("EPSG:3035", "EPSG:4326")

const scalers = Dict(
    :SOCconc   => 0.151, # g/kg, log(x+1)*0.151
    :CF        => 0.263, # percent, log(x+1)*0.263
    :BD        => 0.529, # g/cm3, x*0.529
    :SOCdensity => 0.167, # kg/m3, log(x)*0.167
);

function make_tiles(xs, ys; tilesize = 1024)
    xtiles = Iterators.partition(1:length(xs), tilesize)
    ytiles = Iterators.partition(1:length(ys), tilesize)
    collect(Iterators.product(xtiles, ytiles))
end

function convert_bbox_wgs84_to_3035(bbox_wgs84) 
    xmin_lon, ymin_lat, xmax_lon, ymax_lat = bbox_wgs84 
    tf = Proj.Transformation("EPSG:4326", "EPSG:3035") 
    
    y1, x1 = tf(ymin_lat, xmin_lon) 
    y2, x2 = tf(ymax_lat, xmin_lon) 
    y3, x3 = tf(ymin_lat, xmax_lon) 
    y4, x4 = tf(ymax_lat, xmax_lon) 
    
    xs = (x1, x2, x3, x4) 
    ys = (y1, y2, y3, y4) 
    return (minimum(xs), minimum(ys), maximum(xs), maximum(ys)) 
end

function make_grid_3035(bbox, res_m)
    # bbox = (xmin, ymin, xmax, ymax) in EPSG:3035

    xmin, ymin, xmax, ymax = bbox

    xs = collect(xmin:res_m:xmax)
    ys = collect(ymax:-res_m:ymin)  # north → south

    return xs, ys
end

function sample_tiff_onto_grid(tif_path, xs, ys)
    ArchGDAL.read("/vsicurl/" * tif_path) do ds
        same_crs = occursin("3035", lowercase(ArchGDAL.getproj(ds)))

        band = ArchGDAL.getband(ds, 1)
        x0, dx, _, y0, _, dy = ArchGDAL.getgeotransform(ds)

        nx = length(xs)
        ny = length(ys)
        arr = Matrix{Float32}(undef, ny, nx)

        @inbounds for i in 1:ny, j in 1:nx
            if same_crs
                xr = xs[j]
                yr = ys[i]
            else
                yr, xr = TF_3035_TO_4326(ys[i], xs[j])
            end

            px = Int(round((xr - x0) / dx)) + 1
            py = Int(round((yr - y0) / dy)) + 1

            arr[i, j] =
                (1 ≤ px ≤ ArchGDAL.width(ds) &&
                 1 ≤ py ≤ ArchGDAL.height(ds)) ?
                Float32(ArchGDAL.read(band, py:py, px:px)[1]) :
                NaN32
        end
        vec(arr)
    end
end

const AG = ArchGDAL

function write_geotiff_from_grid(df::DataFrame, value_col::Symbol, filename::String)
    xs = sort(unique(df.x3035))
    ys = sort(unique(df.y3035), rev=true)

    nx, ny = length(xs), length(ys)
    grid = fill(NaN, ny, nx)

    x_index = Dict(x => i for (i, x) in enumerate(xs))
    y_index = Dict(y => i for (i, y) in enumerate(ys))

    for r in eachrow(df)
        grid[y_index[r.y3035], x_index[r.x3035]] = r[value_col]
    end

    dx = xs[2] - xs[1]
    dy = ys[1] - ys[2]

    drv = ArchGDAL.getdriver("GTiff")

    ArchGDAL.create(
        drv;
        filename = filename,
        width = nx,
        height = ny,
        nbands = 1,
        dtype = Float64
    ) do ds
        ArchGDAL.setproj!(ds, "EPSG:3035")
        ArchGDAL.setgeotransform!(
            ds,
            [xs[1] - dx/2, dx, 0.0, ys[1] + dy/2, 0.0, -dy]
        )
        ArchGDAL.write!(ds, permutedims(grid, (2,1)), 1)
        # ArchGDAL.write!(ds, grid, 1)
    end
end

function preprocess_predictors!(
    df::DataFrame,
    predictors::Vector{Symbol},
    cov_scaler::Dict
)
    for col in predictors
        if occursin("_soilsuite_", String(col))
            df[!, col] = replace(df[!, col], missing => 0)
        elseif occursin("cropland_extent_", String(col))
            df[!, col] = replace(df[!, col], missing => 0)
            df[!, col] .= df[!, col] .> 0
        end
    end

    for col in predictors
        μ = cov_scaler[col].mean
        σ = cov_scaler[col].std
        df[!, col] = (Float64.(df[!, col]) .- μ) ./ σ
    end

    return df
end


function load_last_epoch(jldfile)
    jldopen(jldfile, "r") do f
        g = f["HybridModel_SingleNNHybridModel"]
        epochs = sort(
            parse.(Int, replace.(keys(g), "epoch_" => ""))
        )
        last_epoch = "epoch_$(last(epochs))"
        ge = g[last_epoch]
        return ge
    end
end

function SOCD_model(; SOCconc, CF, oBD, mBD)
    ϵ = 1e-7

    # invert transforms
    soct = (exp.(SOCconc ./ scalers[:SOCconc]) .- 1) ./ 1000
    soct = clamp.(soct, ϵ, Inf)
    
    cft = (exp.(CF ./ scalers[:CF]) .- 1) ./ 100
    cft = clamp.(cft, 0, 0.99)

    # compute BD safely
    som = 1.724f0 .* soct
    som = clamp.(som, 0, 1) # test!!!!!!!!
    
    denom = som .* mBD .+ (1f0 .- som) .* oBD
    # denom = clamp.(denom, ϵ, Inf)

    BD = (oBD .* mBD) ./ denom
    BD = clamp.(BD, ϵ, Inf)

    # SOCdensity
    SOCdensity = soct .* 1000 .* BD .* (1 .- cft)
    SOCdensity = clamp.(SOCdensity, 1, Inf)

    # scale
    SOCdensity = log.(SOCdensity) .* scalers[:SOCdensity]
    BD = BD .* scalers[:BD]

    return (; BD, SOCconc, CF, SOCdensity, oBD, mBD)
end


end
