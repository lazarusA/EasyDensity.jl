using Parquet2, Tables, DataFrames
using Statistics
using GLMakie, CairoMakie
GLMakie.activate!()

version = "v20251216"

ds = Parquet2.Dataset("eval/all_cv.pred_with.lc_$(version).pq")
df = DataFrame(ds; copycols=false)

land_covers = [
    "artificial",
    "bareland",
    "cropland",
    "grassland",
    "shrubland",
    "woodland",
    "wetland"
]
targets = ["pred_oBD", "pred_mBD"]

df_oBD = dropmissing(df[:, ["LC_group", "pred_oBD"]])
df_mBD = dropmissing(df[:, ["LC_group", "pred_mBD"]])

df_oBDg = groupby(df_oBD, "LC_group")
df_mBDg = groupby(df_mBD, "LC_group")

colors = repeat([:grey15], length(land_covers))

CairoMakie.activate!() # uncomment this to save pdf files.
mkpath(joinpath(@__DIR__, "../figures/"))

with_theme(theme_latexfonts()) do

        fig = Figure(; size = (1200, 350), fontsize=15)
        ax1 = Axis(fig[1, 1], xticks = (1:length(land_covers), land_covers),
            ylabel = "oBD (g/cm³)", bottomspinecolor = :firebrick, ylabelcolor = :firebrick,
            xlabelsize = 16, ylabelsize=16, xticklabelsize = 16, yticklabelsize=16,)

        for (indx, f) in enumerate(land_covers)
            datam = df_oBDg[("$(f)",)][!, :pred_oBD]
            # filter just in case
            datam = filter(x -> x !== missing, datam)
            datam = replace(datam, missing => NaN)

            a = fill(indx, length(datam))
            boxplot!(ax1, a, datam; whiskerwidth = 0.65, width = 0.35,
                color=:transparent, strokewidth = 1.25, outlierstrokewidth=0.85,
                outlierstrokecolor = (colors[indx], 0.45),
                strokecolor = (colors[indx], 0.85), whiskercolor = (colors[indx], 1),
                # mediancolor = :black,
                mediancolor= :orangered,
                medianlinewidth = 0.85,
                )
        end

        hidespines!(ax1, :l, :r, :t)
        ax2 = Axis(fig[1, 2], xticks = (1:length(land_covers), land_covers),
            ylabel = "mBD (g/cm³)", xtrimspine=true, ytrimspine=true, ylabelcolor=:darkgreen,
            bottomspinecolor = :darkgreen,
            xlabelsize = 16, ylabelsize=16, xticklabelsize = 16, yticklabelsize=16,)

        for (indx, f) in enumerate(land_covers)
            datam = df_mBDg[("$(f)",)][!, :pred_mBD]
            # filter just in case
            datam = filter(x -> x !== missing, datam)
            datam = replace(datam, missing => NaN)

            a = fill(indx, length(datam))
            boxplot!(ax2, a, datam; whiskerwidth = 0.65, width = 0.35,
                color=:transparent, strokewidth = 1.25, outlierstrokewidth=0.85,
                outlierstrokecolor = (colors[indx], 0.45),
                strokecolor = (colors[indx], 0.85), whiskercolor = (colors[indx], 1),
                # mediancolor = :black,
                mediancolor= :darkgreen,
                medianlinewidth = 0.85,
                )
        end
        hidespines!(ax2, :l, :r, :t)
        [Label(fig[1, j, TopLeft()], k,
            fontsize = 22,
            padding = (0, 35, 5, 0),
            halign = :right
            ) for (j, k) in enumerate(["(a)", "(b)"])]
        fig
        save(joinpath(@__DIR__, "../figures/plausibility_oBD_mBD.pdf"), fig)
end