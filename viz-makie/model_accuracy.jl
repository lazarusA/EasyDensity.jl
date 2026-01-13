
using Parquet2, Tables, DataFrames
using Statistics
using GLMakie, CairoMakie
GLMakie.activate!()

#  "CET-L19"
cmap_list = ["#abdda4", "#ffffbf", "#fdae61", "#d7191c"]

version = "v20251216"
targets = ["SOCconc", "CF", "BD", "SOCdensity"]
labels = ["SOC content", "CF", "BD", "SOC density"]
models = ["UniNN", "MultiNN", "SiNN"]

ds = Parquet2.Dataset("eval/all_cv.pred_with.lc_$(version).pq")
df = DataFrame(ds; copycols=false)

function compute_apply_mask(y_pred, y_target)
    mask = .!ismissing.(y_pred) .& .!ismissing.(y_target)
    return replace(y_pred[mask], missing => NaN), replace(y_target[mask], missing => NaN)
end

function compute_r2_mse(y_pred, y_target)
    _mse = mean((y_target .- y_pred).^2)
    denom = sum((y_target .- mean(y_target)).^2)
    _r2 = 1 - sum((y_target .- y_pred).^2)/denom
    _bias = mean(y_pred .- y_target)
    return _r2, _mse, _bias
end

CairoMakie.activate!() # uncomment this to save pdf files.
mkpath(joinpath(@__DIR__, "../figures/"))

with_theme(theme_latexfonts()) do
    for (k, t) in enumerate(targets)
        y_target = df[!, t]
        fig = Figure(; size = (1200, 350), fontsize=15)
        axs = [Axis(fig[1, j], aspect= 1, xlabel = "Prediction", ylabel = "Observation", titlefont = :regular)
            for j in 1:3]
        plt = nothing
        set_upper_count = 1000 # ? we could have different bounds for different targets, but having one to compare among all is good, unless we want to highlight the difference also on the amount of samples per variable. 
        for (i, model) in enumerate(models)
            # @show "$(model)_$t"
            y_pred = df[!, "$(model)_$t"]
            y_pred_new, y_target_new = compute_apply_mask(y_pred, y_target);
            # @show size(y_target_new), size(y_pred_new)
            _r2 , _mse, _bias = compute_r2_mse(y_pred_new, y_target_new)

            plt = hexbin!(axs[i], y_pred_new, y_target_new; cellsize = 0.025, threshold = 1,
                colormap = cmap_list, colorscale=log10,
                colorrange = (1, set_upper_count), highclip = :grey20, lowclip=:transparent)
            # ! one to one line
            lines!(axs[i], [Point2f(0,0), Point2f(1,1)], color = :grey15)
            # ! title
            axs[i].title = rich(rich("$model  ", font=:bold,),
                rich("R", superscript("2")," = $(round(_r2, digits=2))", color=:orangered),
                rich("  MSE", " = $(round(_mse, digits=4))", color=:black),
                rich("\nBias", " = $(round(_bias, digits=4))", color=:dodgerblue),
                )
        end
        # ! label panel
        [Label(fig[1, j, TopLeft()], k,
            fontsize = 22,
            padding = (0, 10, 5, 0),
            halign = :right
            ) for (j, k) in enumerate(["(a)", "(b)", "(c)"])]

        cb = Colorbar(fig[1, 4], plt;
            label = rich(rich("Count", font=:bold), rich(
                "\n\n Cross-validation performance of\n $(labels[k]) predictions."), # for\n other survey years using models trained\n on 2018 data.
                # rich("\n\nTemporal transferability", font=:bold)
                ),
            labelrotation = 0,
            minorticksvisible=true,
            minorticks=IntervalsBetween(9),
            ticks = [1, 10, 100, 1000],
            scale=log10)
        limits!.(axs, 0, 1, 0, 1)
        [axs[j].xticks = (0:0.25:1, ["0", "0.25", "0.5", "0.75", "1"]) for j in 1:3]
        [axs[j].yticks = (0:0.25:1, ["0", "0.25", "0.5", "0.75", "1"]) for j in 1:3]
        hideydecorations!.(axs[2:end], ticks=false, grid=false)
        hidespines!.(axs, :t, :r)
        fig
        save(joinpath(@__DIR__, "../figures/model_accuracy_$(t).pdf"), fig)
    end
end