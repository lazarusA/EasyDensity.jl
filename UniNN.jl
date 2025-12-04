using Pkg
Pkg.activate(".")
using AxisKeys
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
using Flux
using NNlib 
using JLD2

testid = "01_uniNN"
version = "v20251125";
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

@info "Threads: $(Threads.nthreads())"

@time Threads.@threads for test_fold in 1:k
    @info "Fold $test_fold on thread $(Threads.threadid())"

    # -----------------------------
    # Split training / test sets
    # -----------------------------
    train_idx = findall(in(setdiff(1:k, test_fold)), folds)
    test_idx  = findall(==(test_fold), folds)

    train_df = df[train_idx, :]
    test_df_full = df[test_idx, :]

    # Storage for this fold
    fold_params = DataFrame()

    # -----------------------------
    # Loop over single target
    # -----------------------------
    for tgt in target_names
        @info "Target $tgt"

        # ----- train: drop missing -----
        train_df_t = dropmissing(train_df, tgt)
        if nrow(train_df_t) == 0
            @warn "No training rows for $tgt — filling NaN for all rows"
            test_df_full[!, Symbol("pred_", tgt)] = fill(NaN32, nrow(test_df_full))
            continue
        end


        best_loss   = Inf
        best_cfg    = nothing
        best_rlt    = nothing
        best_nn     = nothing

        for h in hidden_configs, bs in batch_sizes, lr in lrs, act in activations

            nn_local = constructNNModel(
                predictors, [tgt];
                hidden_layers = collect(h),
                activation = act,
                scale_nn_outputs = true,
                input_batchnorm = true
            )

            rlt = train(
                nn_local, train_df_t, ();
                nepochs = 200,
                batchsize = bs,
                opt = AdamW(lr),
                training_loss = :mse,
                loss_types = [:mse, :r2],
                shuffleobs = true,
                file_name = "$(tgt)_history_$(testid)_fold$(test_fold).jld2",
                patience = 15,
                return_model = :best,
                plotting = false,
                show_progress = false,
                hybrid_name = "$(tgt)_$(testid)_fold$(test_fold)" 
            )

            if rlt.best_loss < best_loss
                best_loss = rlt.best_loss
                best_cfg  = (h=h, bs=bs, lr=lr, act=act)
                best_rlt  = rlt
                best_nn   = deepcopy(nn_local)
            end
        end

        agg = :sum
        r2s  = map(vh -> getproperty(vh, agg), best_rlt.val_history.r2)
        mses = map(vh -> getproperty(vh, agg), best_rlt.val_history.mse)
        be   = best_rlt.best_epoch

        push!(fold_params, (
            fold       = test_fold,
            target     = String(tgt),
            h          = string(best_cfg.h),
            bs         = best_cfg.bs,
            lr         = best_cfg.lr,
            act        = string(best_cfg.act),
            r2         = r2s[be],
            mse        = mses[be],
            best_epoch = be
        ))

        ps, st = best_rlt.ps, best_rlt.st
        
        try
            # remove missing rows for the current target
            test_df_t = dropmissing(test_df_full, tgt)

            # prepare model input
            x_test, _ = prepare_data(best_nn, test_df_t)
            ŷ, _ = best_nn(x_test, ps, LuxCore.testmode(st))

            preds_clean = ŷ[tgt]  # predictions on filtered rows
            rids_clean  = test_df_t.row_id # row_ids for those rows
            
            pred_df = DataFrame(
                row_id = rids_clean,
                Symbol("pred_", tgt) => preds_clean
            )
            
            test_df_full = leftjoin(
                test_df_full,
                pred_df,
                on = :row_id,
                makeunique = true
            )
            
            replace!(test_df_full[!, Symbol("pred_", tgt)], missing => NaN32)

        catch err
            @warn "Prediction failed for $tgt on fold $test_fold — using NaN"
            test_df_full[!, Symbol("pred_", tgt)] = fill(NaN32, nrow(test_df_full))
        end


    end

    # save results for this fold
    rlt_list_param[test_fold] = fold_params
    rlt_list_pred[test_fold]  = test_df_full
end


# final combined outputs
rlt_param = vcat(rlt_list_param...)
rlt_pred  = vcat(rlt_list_pred...)

CSV.write(joinpath(results_dir, "$(testid)_cv.pred_$version.csv"), rlt_pred)
CSV.write(joinpath(results_dir, "$(testid)_hyperparams_$version.csv"), rlt_param)

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
#     xlims = (0,1.8),
#     ylims = (0, 0.6)
# )   
# savefig(plt, joinpath(results_dir, "$(testid)_BD.vs.SOCconc.png"));
