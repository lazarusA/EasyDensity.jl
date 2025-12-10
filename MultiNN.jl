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

# 02 - multivariate NN
testid = "02_multiNN";
version = "v20251209";
results_dir = joinpath(@__DIR__, "eval");
targets = [:BD, :SOCconc, :SOCdensity, :CF];

# input
df = CSV.read(joinpath(@__DIR__, "data/lucas_preprocessed_v20251125.csv"), DataFrame; normalizenames=true)

# scales
scalers = Dict(
    :SOCconc   => 0.151, # g/kg, log(x+1)*0.151
    :CF        => 0.263, # percent, log(x+1)*0.263
    :BD        => 0.529, # g/cm3, x*0.529
    :SOCdensity => 0.167, # kg/m3, log(x)*0.167
);

# predictor
predictors = Symbol.(names(df))[18:end-6] # CHECK EVERY TIME 
nf = length(predictors)

# configuration
# search space
hidden_configs = [ 
    (512, 256, 128, 64, 32, 16),
    (512, 256, 128, 64, 32), 
    (256, 128, 64, 32, 16),
    (256, 128, 64, 32),
    (256, 128, 64),
    (128, 64, 32, 16),
    (128, 64, 32),
    (64, 32, 16)
];
batch_sizes = [128, 256, 512];
lrs = [1e-3, 5e-4, 1e-4];
activations = [relu, swish, gelu];

configs = [(h=h, bs=bs, lr=lr, act=act)
           for h in hidden_configs
           for bs in batch_sizes
           for lr in lrs
           for act in activations]

println(length(configs))

# cross-validation
k = 5;
folds = make_folds(df, k = k, shuffle = true);
rlt_list_param = Vector{DataFrame}(undef, k);
rlt_list_pred = Vector{DataFrame}(undef, k);  
@info "Threads available: $(Threads.nthreads())"

@time for test_fold in 1:k
    @info "Training outer fold $test_fold of $k"

    train_folds = setdiff(1:k, test_fold)
    train_idx = findall(in(train_folds), folds)
    train_df = df[train_idx, :]
    test_idx  = findall(==(test_fold), folds)
    test_df = df[test_idx, :]
    
    # track best config for this outer fold
    lk = ReentrantLock()
    best_val_loss = Inf
    best_config = nothing
    best_result = nothing
    best_model = nothing
    best_model_path = nothing
    results_param = DataFrame(h=String[], bs=Int[], lr=Float64[], act=String[], r2=Float64[], mse=Float64[], best_epoch=Int[], test_fold=Int[])

    # param search by looping....    
    Threads.@threads for i in 1:length(configs)
        try
            cfg = configs[i]
        
            h  = cfg.h
            bs = cfg.bs
            lr = cfg.lr
            act = cfg.act
            println("Testing h=$h, bs=$bs, lr=$lr, activation=$act")
        
            nn_local = EasyHybrid.constructNNModel(
                predictors, targets;
                hidden_layers = collect(h),
                activation = act,
                scale_nn_outputs = true,
                input_batchnorm = false
            )
            
            rlt = train(
                nn_local, train_df, ();
                nepochs = 200,
                batchsize = bs,
                opt = AdamW(lr),
                training_loss = :mse,
                loss_types = [:mse, :r2],
                shuffleobs = true,
                file_name = "$(testid)_config$(i)_fold$(test_fold).jld2",
                random_seed = 42,
                patience = 15,
                yscale = identity,
                agg = mean,
                return_model = :best,
                show_progress = false,
                plotting = false,
                hybrid_name = "$(testid)_config$(i)_fold$(test_fold)"
            )
    
            lock(lk)
            if rlt.best_loss < best_val_loss
                best_val_loss = rlt.best_loss
                best_config = cfg
                best_result = rlt
                best_model = deepcopy(nn_local)
                best_model_path = "best_model_$(testid)_config$(i)_fold$(test_fold)"
            end
            unlock(lk)
        catch err
            @error "Thread $i crashed" exception = err
            @error sprint(showerror, err)
        end
    end

    # register best hyper paramets
    agg_name = Symbol("mean")
    r2s  = map(vh -> getproperty(vh, agg_name), best_result.val_history.r2)
    mses = map(vh -> getproperty(vh, agg_name), best_result.val_history.mse)
    best_epoch = max(best_result.best_epoch, 1)

    local_results_param = DataFrame(
        h = string(best_config.h),
        bs = best_config.bs,
        lr = best_config.lr,
        act = string(best_config.act),
        r2 = r2s[best_epoch],
        mse = mses[best_epoch],
        best_epoch = best_epoch,
        test_fold = test_fold,
        path = best_model_path,
    )
    rlt_list_param[test_fold] = local_results_param

    # move best models and then remove tmp files
    cp(joinpath("output_tmp", best_model_path * ".jld2"), joinpath("model", best_model_path * ".jld2"); force=true) 
    for f in readdir("output_tmp"; join=true)
        rm(f; force=true, recursive=true)
    end
    

    (x_test,  y_test)  = prepare_data(best_model, test_df)
    ps, st = best_result.ps, best_result.st
    ŷ_test, st_test = best_model(x_test, ps, LuxCore.testmode(st))
    # println(propertynames(ŷ_test))

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

