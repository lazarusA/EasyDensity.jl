using Pkg
# Pkg.activate(".")
# Pkg.instantiate()
using Revise
using EasyHybrid
using Lux
using Optimisers
using WGLMakie
using Random
using LuxCore
using CSV, DataFrames
using EasyHybrid.MLUtils
using Statistics
using Plots
using JLD2
# using CairoMakie

# 03 - flexiable BD, both oBD and mBD will be learnt by NN
testid = "03_hybridNN";
results_dir = joinpath(@__DIR__, "eval");
target_names = [:BD, :SOCconc, :CF, :SOCdensity];

# input
df = CSV.read(joinpath(@__DIR__, "data/lucas_preprocessed_v20251113.csv"), DataFrame; normalizenames=true)

# scales
scalers = Dict(
    :SOCconc   => 0.151, # g/kg, log(x+1)*0.151
    :CF        => 0.263, # percent, log(x+1)*0.263
    :BD        => 0.529, # g/cm3, x*0.529
    :SOCdensity => 0.167, # kg/m3, log(x)*0.167
);

for tgt in target_names
    # println(tgt, "------------")
    # println(minimum(df[:,tgt]), "  ", maximum(df[:,tgt]))
    if tgt in (:SOCconc, :CF)
        df[!, tgt] .= log.(df[!, tgt] .+ 1) 
        # println(minimum(df[:,tgt]), "  ", maximum(df[:,tgt]))
    elseif tgt == :SOCdensity
        df[!, tgt] .= log.(df[!, tgt]) 
    end

    df[!, tgt] .= df[!, tgt] .* scalers[tgt]
    # println(minimum(df[:,tgt]), "  ", maximum(df[:,tgt]))
end

for col in ["BD", "SOCconc", "CF", "SOCdensity"]
    # values = log10.(df[:, col])
    values = df[:, col]
    histogram(
        values;
        bins = 50,
        xlabel = col,
        ylabel = "Frequency",
        title = "Histogram of $col",
        lw = 1,
        legend = false
    )
    display(current())
end

# mechanistic model
function SOCD_model(; SOCconc, CF, oBD, mBD)
    soct = (exp.(SOCconc ./ scalers[:SOCconc]) .- 1) ./ 1000 # to fraction
    cft = (exp.(CF ./ scalers[:CF]) .- 1) ./ 100  # back to fraction
    BD = (oBD .* mBD) ./ (1.724f0 .* soct .* mBD .+ (1f0 .- 1.724f0 .* soct) .* oBD)
    SOCdensity = soct .*1000 .* BD .* (1 .- cft) # kg/m3
    
    SOCdensity = log.(SOCdensity) .* scalers[:SOCdensity]  # scale to ~[0,1]
    BD = BD .* scalers[:BD]  # scale to ~[0,1]
    return (; BD, SOCconc, CF, SOCdensity, oBD, mBD)  # supervise both BD and SOCconc
end

# param bounds
parameters = (
    SOCconc = (0.01f0, 0.0f0, 1.0f0),   # fraction
    CF      = (0.15f0, 0.0f0, 1.0f0),   # fraction,
    oBD     = (0.20f0, 0.05f0, 0.40f0),  # also NN learnt, g/cm3
    mBD     = (1.20f0, 0.75f0, 2.0f0),  # NN leanrt
)

# define param for hybrid model
neural_param_names = [:SOCconc, :CF, :mBD, :oBD]
# global_param_names = [:oBD]
forcing = Symbol[]
targets = [:BD, :SOCconc, :SOCdensity, :CF]       # SOCconc is both a param and a target

# predictor
predictors = Symbol.(names(df))[19:end-2]; # CHECK EVERY TIME 
nf = length(predictors)

# search space
hidden_configs = [ 
    (512, 256, 128, 64, 32, 16),
    (512, 256, 128, 64, 32), 
    (256, 128, 64, 32, 16),
    (256, 128, 64, 32),
    (256, 128, 64),
    (128, 64, 32, 16),
    (128, 64, 32),
    (128, 64),
    (64, 32, 16)
];
batch_sizes = [128, 256, 512];
lrs = [1e-3, 5e-4, 1e-4];
activations = [relu, tanh, swish, gelu];


# cross-validation
k = 5;
folds = make_folds(df, k = k, shuffle = true);
rlt_list_param = Vector{DataFrame}(undef, k)
rlt_list_pred = Vector{DataFrame}(undef, k)  

@info "Threads available: $(Threads.nthreads())"
@time Threads.@threads for test_fold in 1:k
    @info "Training outer fold $test_fold of $k on thread $(Threads.threadid())"

    train_folds = setdiff(1:k, test_fold)
    train_idx = findall(in(train_folds), folds)
    train_df = df[train_idx, :]
    test_idx  = findall(==(test_fold), folds)
    test_df = df[test_idx, :]

    # track best config for this outer fold
    best_val_loss = Inf
    best_config = nothing
    best_result = nothing
    best_hm = nothing
    results_param = DataFrame(h=String[], bs=Int[], lr=Float64[], act=String[], r2=Float64[], mse=Float64[], best_epoch=Int[], test_fold=Int[])

    for h in hidden_configs, bs in batch_sizes, lr in lrs, act in activations
        println("Testing h=$h, bs=$bs, lr=$lr, activation=$act")

        hm_local = constructHybridModel(
            predictors,
            forcing,
            targets,
            SOCD_model,
            parameters,
            neural_param_names,
            [];
            hidden_layers = collect(h),
            activation = act,
            scale_nn_outputs = true,
            input_batchnorm = true,
            start_from_default = true
        )

        rlt = train(
            hm_local, train_df, ();
            nepochs = 200,
            batchsize = bs,
            opt = AdamW(lr),
            training_loss = :mse,
            loss_types = [:mse, :r2],
            shuffleobs = true,
            file_name = "history_$(testid)_fold$(test_fold).jld2",
            random_seed = 42,
            patience = 15,
            yscale = identity,
            monitor_names = [:oBD, :mBD],
            agg = mean,
            return_model = :best,
            show_progress = false,
            plotting = false,
            hybrid_name = "$(testid)_fold$(test_fold)" 
        )

        if rlt.best_loss < best_val_loss
            best_config = (h=h, bs=bs, lr=lr, act=act)
            best_result = rlt
            best_hm = deepcopy(hm_local)
        end
    end

    # register best hyper paramets
    agg_name = Symbol("mean")
    r2s  = map(vh -> getproperty(vh, agg_name), best_result.val_history.r2)
    mses = map(vh -> getproperty(vh, agg_name), best_result.val_history.mse)
    best_epoch = best_result.best_epoch


    local_results_param = DataFrame(
        h = string(best_config.h),
        bs = best_config.bs,
        lr = best_config.lr,
        act = string(best_config.act),
        r2 = r2s[best_epoch],
        mse = mses[best_epoch],
        best_epoch = best_epoch,
        test_fold = test_fold
    )
    rlt_list_param[test_fold] = local_results_param
    

    (x_test,  y_test)  = prepare_data(best_hm, test_df)
    ps, st = best_result.ps, best_result.st
    ŷ_test, st_test = best_hm(x_test, ps, LuxCore.testmode(st))
    println(propertynames(ŷ_test))
    println(propertynames(ŷ_test.parameters))

    for var in [:BD, :SOCconc, :CF, :SOCdensity, :oBD, :mBD]
        if hasproperty(ŷ_test, var)
            val = getproperty(ŷ_test, var)

            if val isa AbstractVector && length(val) == nrow(test_df)
                test_df[!, Symbol("pred_", var)] = val # per row

            elseif (val isa Number) || (val isa AbstractVector && length(val) == 1)
                test_df[!, Symbol("pred_", var)] = fill(Float32(val isa AbstractVector ? first(val) : val), nrow(test_df))
            end


        end
    end
    
    rlt_list_pred[test_fold] = test_df

end

rlt_param = vcat(rlt_list_param...)
rlt_pred = vcat(rlt_list_pred...)

CSV.write(joinpath(results_dir, "$(testid)_cv.pred_v20251113.csv"), rlt_pred)
CSV.write(joinpath(results_dir, "$(testid)_hyperparams_v20251113.csv"), rlt_param)

# # print best model
# @assert best_bundle !== nothing "No valid model found for $testid"
# bm = best_bundle
# file_path = joinpath(results_dir, "$(testid)_best_model.jld2")
# jldsave(file_path;
#     ps=best_bundle.ps, st=best_bundle.st, model=best_bundle.model,
#     val_obs_pred=best_bundle.val_obs_pred, val_diffs=best_bundle.val_diffs,
#     meta=best_bundle.meta,
#     mBD_phys=best_bundle.mBD_phys,
#     oBD_physical=best_bundle.oBD_physical,      # use the actual field
# )

# # @load joinpath(results_dir, "best_model_$(tgt).jld2") ps st model val_obs_pred meta
# @info "Best for $testid: bs=$(bm.meta.bs), lr=$(bm.meta.lr), act=$(bm.meta.act), epoch=$(bm.meta.best_epoch), R2=$(round(best_r2, digits=4))"

# # load predictions
# jld = joinpath(results_dir, "$(testid)_best_model.jld2")
# @assert isfile(jld) "Missing $(jld). Did you train & save best model for $(tname)?"
# @load jld val_obs_pred meta
# # split output table
# val_tables = Dict{Symbol,Vector{Float64}}()
# for t in targets
#     # expected: t (true), t_pred (pred), and maybe :index if the framework saved it
#     have_pred = Symbol(t, :_pred)
#     req = Set((t, have_pred))
#     @assert issubset(req, Symbol.(names(val_obs_pred))) "val_obs_pred missing $(collect(req)) for $(t). Columns: $(names(val_obs_pred))"
#     val_tables[t] = val_obs_pred[:, t]./ scalers[t]
#     val_tables[have_pred] = val_obs_pred[:, have_pred]./ scalers[t]
#     if t in (:SOCdensity, :SOCconc)
#         val_tables[Symbol("$(t)_pred")] = exp.(val_tables[Symbol("$(t)_pred")]) ./ 1000
#         val_tables[t] = exp.(val_tables[t]) ./ 1000
#     end
# end


# # helper for metrics calculation
# r2_mse(y_true, y_pred) = begin
#     ss_res = sum((y_true .- y_pred).^2)
#     ss_tot = sum((y_true .- mean(y_true)).^2)
#     r2  = 1 - ss_res / ss_tot
#     mse = mean((y_true .- y_pred).^2)
#     (r2, mse)
# end

# # accuracy plots for SOCconc, BD, CF in original space
# for tname in targets
#     y_val_true = val_tables[tname]
#     y_val_pred = val_tables[Symbol("$(tname)_pred")]

#     # @assert all(in(Symbol.(names(df_out))).([tname, Symbol("$(tname)_pred")])) "Expected columns $(tname) and $(tname)_pred in saved val table."

#     r2, mse = r2_mse(y_val_true, y_val_pred)

#     plt = histogram2d(
#         y_val_pred, y_val_true;
#         nbins=(40, 40), cbar=true, xlab="Predicted", ylab="Observed",
#         title = string(tname, "\nR²=", round(r2, digits=3), ", MSE=", round(mse, digits=3)),
#         normalize=false
#     )
#     lims = extrema(vcat(y_val_true, y_val_pred))
#     Plots.plot!(plt, [lims[1], lims[2]], [lims[1], lims[2]];
#         color=:black, linewidth=2, label="1:1 line",
#         aspect_ratio=:equal, xlims=lims, ylims=lims
#     )
#     savefig(plt, joinpath(results_dir, "$(testid)_accuracy_$(tname).png"))
# end

# # BD vs SOCconc predictions
# plt = histogram2d(
#     val_tables[:BD_pred], val_tables[:SOCconc_pred];
#     nbins      = (30, 30),
#     cbar       = true,
#     xlab       = "BD",
#     ylab       = "SOCconc",
#     color      = cgrad(:bamako, rev=true),
#     normalize  = false,
#     size = (460, 400),
#     xlims     = (0, 1.8),
#     ylims     = (0, 0.6)
# )   
# savefig(plt, joinpath(results_dir, "$(testid)_BD.vs.SOCconc.png"));

# # save / print parameters: mBD and oBD
# @load jld oBD_physical
# histogram(oBD_physical; bins=:sturges, xlabel="learned oBD", ylabel="count",
#           title="Distribution of learned oBD", legend=false)
# vline!([mean(oBD_physical)]; lw=2, label=false)  # mean marker

# @load jld mBD_phys
# histogram(mBD_phys; bins=:sturges, xlabel="learned mBD", ylabel="count",
#           title="Distribution of learned mBD", legend=false)
# vline!([mean(mBD_phys)]; lw=2, label=false)  # mean marker
# @info "Saved histogram to $(joinpath(results_dir, "$(testid)_mBD_histogram.png"))"

# # plot mBD_phys and texture
# texture = CSV.read(joinpath(@__DIR__, "data/lucas_test_texture.csv"), DataFrame; normalizenames=true)
# texture.mBD_phys = mBD_phys
# texture.oBD_phys = oBD_physical


# for col in (:clay, :silt, :sand)
#     p = Plots.scatter(texture[!, col], texture.BD; label="BD", xlabel=String(col), ylabel="Density", legend=:topleft)
#     Plots.scatter!(texture[!, col], texture.mBD_phys; label="mBD_phys")
#     title!(p, "$(col) vs density")
#     display(p)
#     # savefig("$(col)_vs_density.png")   # or display without saving
# end

# # drop rows with missings in these columns
# tex = dropmissing(texture, [:clay, :sand, :BD, :mBD_phys])

# # BD plot
# p1 = Plots.scatter(tex.clay, tex.sand;
#     zcolor = tex.BD,
#     xlabel = "Clay",
#     ylabel = "Sand",
#     # colorbar_title = "BD",
#     title = "observed BD",
#     legend = false,
#     colorbar = true,
#     color = cgrad(:algae),
#     # clim = (cmin, cmax),
#     markersize = 2.5, markerstrokewidth = 0,
#     aspect_ratio = :equal)

# # mBD_phys plot
# p2 = Plots.scatter(tex.clay, tex.sand;
#     zcolor = tex.mBD_phys,
#     xlabel = "Clay",
#     ylabel = "Sand",
#     # colorbar_title = "mBD_phys",
#     title = "mineral BD",
#     legend = false,
#     colorbar = true,
#     color = cgrad(:dense),
#     # clim = (cmin, cmax),
#     markersize = 2.5, markerstrokewidth = 0,
#     aspect_ratio = :equal)

# # oBD_phys
# p3 = Plots.scatter(tex.clay, tex.sand;
#     zcolor = tex.oBD_phys,
#     xlabel = "Clay",
#     ylabel = "Sand",
#     # colorbar_title = "mBD_phys",
#     title = "organic BD",
#     legend = false,
#     colorbar = true,
#     color = cgrad(:solar, rev=true),
#     # clim = (cmin, cmax),
#     markersize = 2.5, markerstrokewidth = 0,
#     aspect_ratio = :equal)

# finalplot = Plots.plot(p1, p2, p3, layout=(1,3), size=(1350,450))
# display(finalplot)
# savefig(finalplot, joinpath(results_dir, "$(testid)_texture.vs.BD.png")) 

# # 2D histograms
# p1 = histogram2d(tex.clay, tex.BD; xlabel="Clay (%)", ylabel="Observed BD",
#     bins=40, color=cgrad(:thermal, rev=true), legend=false)
# p2 = histogram2d(tex.clay, tex.mBD_phys; xlabel="Clay (%)", ylabel="Mineral BD",
#     bins=40, color=cgrad(:thermal, rev=true), legend=false)
# p3 = histogram2d(tex.clay, tex.oBD_phys; xlabel="Clay (%)", ylabel="Organic BD",
#     bins=40, color=cgrad(:thermal, rev=true), legend=false)

# # combine and add simple padding around edges
# finalplot = Plots.plot(p1, p2, p3;
#     layout = (1,3),
#     size = (1500, 500),
#     margin = 7Plots.mm)   # ← simplest way to add breathing room

# display(finalplot)
# savefig(finalplot, joinpath(results_dir, "$(testid)_clay.vs.BD.png")) 

# # MTD SOCdensity
# socdensity_pred = val_tables[:SOCconc_pred] .* val_tables[:BD_pred] .* (1 .- val_tables[:CF_pred]);
# socdensity_true = val_tables[:SOCdensity];
# r2_sd, mse_sd = r2_mse(socdensity_true, socdensity_pred);
# plt = histogram2d(
#     socdensity_pred, socdensity_true;
#     nbins=(40,40), cbar=true, xlab="Pred SOCdensity MTD", ylab="True SOCdensity",
#     title = "SOCdensity\nR²=$(round(r2_sd,digits=3)), MSE=$(round(mse_sd,digits=3))",
#     normalize=false
# )
# lims = extrema(vcat(socdensity_true, socdensity_pred))
# Plots.plot!(plt, [lims[1], lims[2]], [lims[1], lims[2]];
#     color=:black, linewidth=2, label="1:1 line",
#     aspect_ratio=:equal, xlims=lims, ylims=lims
# )
# savefig(plt, joinpath(results_dir, "$(testid)_accuracy_SOCdensity.MTD.png"));


