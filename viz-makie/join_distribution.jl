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

# ! filter rows where `bd` and `soc` are non-missing
df = dropmissing(df, [:bd, :soc])

CairoMakie.activate!() # uncomment this to save pdf files.
mkpath(joinpath(@__DIR__, "../figures/"))

with_theme(theme_latexfonts()) do

        fig = Figure(; size = (1200, 350), fontsize=15)
        axs = [Axis(fig[1, j], aspect= 1, xlabel = "BD (g/cm3)", ylabel = "SOC content (g/kg)",
            xlabelsize = 16, ylabelsize=16, xticklabelsize = 16, yticklabelsize=16,
            titlefont = :regular)
            for j in 1:4]
        plt = nothing
        set_upper_count = 1000
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
            ticks = [1, 10, 100, 1000],
            scale=log10)
        limits!.(axs, 0, 2, 0, 625)
        hideydecorations!.(axs[2:end], ticks=false, grid=false)
        hidespines!.(axs, :t, :r)
        fig
        save(joinpath(@__DIR__, "../figures/joint_distribution.pdf"), fig)
end

# do a normal density heatmap!
using AlgebraOfGraphics

with_theme(theme_latexfonts()) do

        fig = Figure(; size = (1200, 350), fontsize=15)
        axs = [Axis(fig[1, j], aspect= 1, xlabel = "BD (g/cm3)", ylabel = "SOC content (g/kg)", titlefont = :regular)
            for j in 1:4]
        plt_aog = nothing
        set_upper_count = 10000 
        for (i, model_combo) in enumerate(xy_to)
            @show model_combo
            y_x = df[!, "$(model_combo[1])"]
            y_y = df[!, "$(model_combo[2])"]

            y_x_new, y_y_new = compute_apply_mask(y_x, y_y);
            @show size(y_x_new), size(y_y_new)

            df_aog = (x=y_x_new, y=y_y_new)
            plt = data(df_aog) * mapping(:x, :y) * histogram(bins=(30, 30))
            plt_aog = draw!(axs[i], plt, scales(Color = (; colormap = cmap_list, scale=log10, colorrange = (1, set_upper_count), highclip = :grey20, lowclip=:transparent )))
            axs[i].title = titles[i]
        end
        # ! label panel
        [Label(fig[1, j, TopLeft()], k,
            fontsize = 22,
            padding = (0, 10, -30, 0),
            halign = :right
            ) for (j, k) in enumerate(["(a)", "(b)", "(c)", "(d)"])]
        # @show plt_aog[1].axis.plt
        cb = Colorbar(fig[1, 5], limits = (1, set_upper_count),
            colormap = cmap_list,
            label = rich(rich("Count", font=:bold),
                ),
            # labelrotation = 0,
            minorticksvisible=true,
            minorticks=IntervalsBetween(9),
            ticks = [1, 10, 100, 1000, 10_000],
             highclip = :grey20, lowclip=:transparent,
            scale=log10)
        limits!.(axs, 0, 2, 0, 625)
        hideydecorations!.(axs[2:end], ticks=false, grid=false)
        hidespines!.(axs, :t, :r)
        fig
        save(joinpath(@__DIR__, "../figures/joint_distribution_density_heatmap.pdf"), fig)
end


# ! check numbers
# variables to check (same as python)
vars_to_check = ["UniNN_ocd", "MultiNN_ocd", "SiNN_ocd"]

df_filtered = copy(df)

# ---- 1. COUNT IDS BY NUMBER OF TIME POINTS ----
id_counts = combine(groupby(df_filtered, :id), nrow => :count)

num_ids_all3  = count(==(3), id_counts.count)
num_ids_not3  = count(!=(3), id_counts.count)

println("IDs with all 3 time points: ", num_ids_all3)
println("IDs WITHOUT all 3 time points: ", num_ids_not3)

# ---- 2. KEEP ONLY IDS THAT HAVE ALL 3 ROWS ----
valid_ids = id_counts.id[id_counts.count .== 3]

df3 = filter(row -> row.id in valid_ids, df_filtered)

println("Filtered dataframe shape: ", size(df3))

# ---- 3. TEMPORAL STABILITY USING IQR AND RANGE ----
# Compute min and max per id for each variable
stability = combine(
    groupby(df3, :id),
    [
        var => minimum => Symbol(var * "_min") for var in vars_to_check
    ]...,
    [
        var => maximum => Symbol(var * "_max") for var in vars_to_check
    ]...
)

# Compute range (max - min)
for var in vars_to_check
    stability[!, Symbol(var * "_range")] =
        stability[!, Symbol(var * "_max")] .-
        stability[!, Symbol(var * "_min")]
end

val_labels = ["UniNN","MultiNN","SiNN"]

for ii in val_labels
    col = skipmissing(stability[!, Symbol(ii * "_ocd_range")])

    q05 = quantile(col, 0.05)
    med = median(col)
    q95 = quantile(col, 0.95)

    println(ii, ": ", q05, "  ", med, "  ", q95)
end

# ! this is my output
# UniNN: 0.76481805617236  3.6588506761590427  12.348510392006892
# MultiNN: 0.6328896287047178  3.02064480793726  10.658227136384529
# SiNN: 0.738088003073246  3.4789800338796812  10.386650944633367