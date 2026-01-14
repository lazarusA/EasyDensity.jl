using Parquet2, Tables, DataFrames
using Statistics
using GLMakie, CairoMakie
GLMakie.activate!()

#  "CET-L19"
cmap_list = ["#abdda4", "#ffffbf", "#fdae61", "#d7191c"]

version = "v20251216"
# targets = ["SOCconc", "CF", "BD", "SOCdensity"]
# labels = ["SOC content", "CF", "BD", "SOC density"]
to_join_dist = ["bd", "soc"]
models = ["", "UniNN_", "MultiNN_", "SiNN_"]
models_raw = ["UniNN", "MultiNN", "SiNN"]

xy_to = []
for m in models
    push!(xy_to, m .* to_join_dist)
end

ds = Parquet2.Dataset("eval/all_cv.pred_with.lc_$(version).pq")
df = DataFrame(ds; copycols=false)

# ? why, from where are these coming from?
scalers = Dict(
    "SOCconc"=> 0.151,
    "CF" => 0.263,
    "BD"=> 0.529,
    "SOCdensity"=> 0.167)

# append new columns
for mod in models_raw
    df[!, "$(mod)_soc"] = @. exp(df[!, "$(mod)_SOCconc"] / scalers["SOCconc"]) - 1
    df[!, "$(mod)_cf"]  = @. exp(df[!, "$(mod)_CF"] / scalers["CF"]) - 1
    df[!, "$(mod)_bd"]  = @. df[!, "$(mod)_BD"] / scalers["BD"] # ? are we getting the same numbers here!
    df[!, "$(mod)_ocd"] = @. exp(df[!, "$(mod)_SOCdensity"] / scalers["SOCdensity"])
end
md_preds = rich.(rich.(models_raw, font=:bold), " prediction")
titles = ["Observation", md_preds...]

function compute_apply_mask(y_pred, y_target)
    mask = .!ismissing.(y_pred) .& .!ismissing.(y_target)
    return replace(y_pred[mask], missing => NaN), replace(y_target[mask], missing => NaN)
end

CairoMakie.activate!() # uncomment this to save pdf files.
mkpath(joinpath(@__DIR__, "../figures/"))

# ! filter outliers !
df = subset(
    df,
    :bd  => ByRow(<(0.2)),
    :soc => ByRow(>(200));
    skipmissing = true
)

with_theme(theme_latexfonts()) do

        fig = Figure(; size = (1200, 350), fontsize=15)
        axs = [Axis(fig[1, j], aspect= 1, xlabel = "BD (g/cm3)", ylabel = "SOC content (g/kg)",
            xlabelsize = 16, ylabelsize=16, xticklabelsize = 16, yticklabelsize=16,
            titlefont = :regular)
            for j in 1:4]
        plt = nothing
        set_upper_count = 500 
        for (i, model_combo) in enumerate(xy_to)
            @show model_combo
            y_x = df[!, "$(model_combo[1])"]
            y_y = df[!, "$(model_combo[2])"]

            y_x_new, y_y_new = compute_apply_mask(y_x, y_y);
            @show size(y_x_new), size(y_y_new)

            plt = hexbin!(axs[i], y_x_new, y_y_new; cellsize = (0.1, 33), threshold = 1,
                colormap = cmap_list, colorscale=log10,
                colorrange = (1, set_upper_count), highclip = :grey20, lowclip=:transparent)

            axs[i].title = titles[i]
        end
        # ! label panel
        [Label(fig[1, j, TopLeft()], k,
            fontsize = 22,
            padding = (0, 10, -30, 0),
            halign = :right
            ) for (j, k) in enumerate(["(a)", "(b)", "(c)", "(d)"])]

        cb = Colorbar(fig[1, 5], plt;
            label = rich(rich("Count", font=:bold),
                ),
            # labelrotation = 0,
            minorticksvisible=true,
            minorticks=IntervalsBetween(9),
            ticks = [1, 10, 100, 500],
            scale=log10)
        limits!.(axs, 0, 2, 0, 625)
        hideydecorations!.(axs[2:end], ticks=false, grid=false)
        hidespines!.(axs, :t, :r)
        fig
        save(joinpath(@__DIR__, "../figures/joint_distribution_outliers.pdf"), fig)
end