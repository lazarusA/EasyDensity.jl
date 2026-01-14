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
    df[!, "$(mod)_bd"]  = @. df[!, "$(mod)_BD"] / scalers["BD"]
    df[!, "$(mod)_ocd"] = @. exp(df[!, "$(mod)_SOCdensity"] / scalers["SOCdensity"])
end

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

[stability[!, "$(v)_range"] for v in vars_to_check]

colors = categorical_colors(:Set1, length(vars_to_check))
colors = repeat([:grey15], 3)

CairoMakie.activate!() # uncomment this to save pdf files.
mkpath(joinpath(@__DIR__, "../figures/"))

with_theme(theme_latexfonts()) do

        fig = Figure(; size = (400, 350), fontsize=15)
        ax = Axis(fig[1, 1], xticks = (1:length(val_labels), rich.(val_labels, font=:bold)),
            ylabel = "SOC density (kg/m3)",
            xlabelsize = 16, ylabelsize=16, xticklabelsize = 16, yticklabelsize=16,)

        for (indx, f) in enumerate(vars_to_check)
            datam = filter(x -> x !== missing, stability[:, "$(f)_range"])
            datam = replace(datam, missing => NaN)
            a = fill(indx, length(datam))
            boxplot!(ax, a, datam; whiskerwidth = 0.65, width = 0.35,
                color=:transparent, strokewidth = 1.25, outlierstrokewidth=0.85,
                outlierstrokecolor = (colors[indx], 0.45),
                strokecolor = (colors[indx], 0.85), whiskercolor = (colors[indx], 1),
                # mediancolor = :black,
                mediancolor= :orangered,
                medianlinewidth = 0.85,
                )
            ax.yticks = 0:20:160
        end
        fig
        save(joinpath(@__DIR__, "../figures/temporal_plausibility_1.pdf"), fig)
end


with_theme(theme_latexfonts()) do

        fig = Figure(; size = (900, 350), fontsize=15)
        ax1 = Axis(fig[1, 1], xticks = (1:length(val_labels), rich.(val_labels, font=:bold)),
            ylabel = "SOC density (kg/m3)", xlabelsize = 16, ylabelsize=16, xticklabelsize = 16, yticklabelsize=16,)
        ax2 = Axis(fig[1, 2], xticks = (1:length(val_labels), rich.(val_labels, font=:bold)),
            xlabelsize = 16, ylabelsize=16, xticklabelsize = 16, yticklabelsize=16,
            )

        for (indx, f) in enumerate(vars_to_check)
            datam = filter(x -> x !== missing, stability[:, "$(f)_range"])
            datam = replace(datam, missing => NaN)
            a = fill(indx, length(datam))
            boxplot!(ax1, a, datam; whiskerwidth = 0.65, width = 0.35,
                color=:transparent, strokewidth = 1.25, outlierstrokewidth=1.25,
                outlierstrokecolor = (colors[indx], 0.45),
                strokecolor = (colors[indx], 0.85), whiskercolor = (colors[indx], 1),
                # mediancolor = :black,
                mediancolor= :orangered,
                medianlinewidth = 0.85,
                markersize = 5,
                )
        end
        ax1.yticks = 0:20:160

        for (indx, f) in enumerate(vars_to_check)
            datam = filter(x -> x !== missing, stability[:, "$(f)_range"])
            datam = replace(datam, missing => NaN)
            a = fill(indx, length(datam))
            boxplot!(ax2, a, datam; whiskerwidth = 0.65, width = 0.35,
                color=:transparent, strokewidth = 1.25, outlierstrokewidth=0.85,
                outlierstrokecolor = (colors[indx], 0.45),
                strokecolor = (colors[indx], 0.85), whiskercolor = (colors[indx], 1),
                # mediancolor = :black,
                mediancolor= :orangered,
                medianlinewidth = 0.85,
                markersize = 5
                )
            # ax2.yticks = 0:20:160
        end
        ax2.yticks = [0, 5, 10, 11, 12]

        ylims!(ax2, -1, 13)
        hidespines!.([ax1, ax2], :t, :r)
        hidespines!(ax2, :l)
        colgap!(fig.layout, 50)
        fig
        save(joinpath(@__DIR__, "../figures/temporal_plausibility_2.pdf"), fig)
end