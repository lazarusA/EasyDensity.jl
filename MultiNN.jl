using Pkg
# Pkg.activate(".")
using Revise
using EasyHybrid
using Lux
using Pkg
Pkg.activate("projects/BulkDSOC")
Pkg.develop(path=pwd())
Pkg.instantiate()

using Revise
using EasyHybrid
using Lux
using Optimisers
using GLMakie
using Random
using LuxCore
using CSV, DataFrames
using EasyHybrid.MLUtils
using Statistics
using Plots
using JLD2

# 03 - sparse data
testid = "02_multiNN";
version = "v20251120";

results_dir = joinpath(@__DIR__, "eval");
target_names = [:BD, :SOCconc, :CF, :SOCdensity];

# input
df = CSV.read(joinpath(@__DIR__, "data/lucas_preprocessed_$version.csv"), DataFrame; normalizenames=true)

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

# predictor
predictors = Symbol.(names(df))[19:end-2] # CHECK EVERY TIME 
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

    # param search by looping....    
    for h in hidden_configs, bs in batch_sizes, lr in lrs, act in activations
        println("Testing h=$h, bs=$bs, lr=$lr, activation=$act")
        
        nn_local = EasyHybrid.constructNNModel(
            predictors, targets;
            hidden_layers = collect(h),
            activation = act,
            scale_nn_outputs = true,
            input_batchnorm = true
        )
        
        rlt = train(
            nn_local, train_df, ();
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

    for var in [:BD, :SOCconc, :CF, :SOCdensity]
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

CSV.write(joinpath(results_dir, "$(testid)_cv.pred_$version.csv"), rlt_pred)
CSV.write(joinpath(results_dir, "$(testid)_hyperparams_$version.csv"), rlt_param)


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
