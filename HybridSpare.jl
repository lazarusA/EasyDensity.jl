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

# 05 -  hybrid and sparse
testid = "05_hybridspase";
results_dir = joinpath(@__DIR__, "eval");

# input
train_df = CSV.read(joinpath(@__DIR__, "data/lucas_train.csv"), DataFrame; normalizenames=true)
# train_df = dropmissing(train_df)
test_df = CSV.read(joinpath(@__DIR__, "data/lucas_test.csv"), DataFrame; normalizenames=true)
test_df = dropmissing(test_df)

# scales
scalers = Dict(
    :SOCconc   => 0.158, # log(x*1000)*0.158
    :CF        => 2.2,
    :BD        => 0.53,
    :SOCdensity => 0.165, # log(x*1000)*0.165
);
for tgt in [:BD, :CF, :SOCconc, :SOCdensity]
    if tgt in (:SOCdensity, :SOCconc)
        train_df[!, tgt] .= log.(train_df[!, tgt] .* 1000)
        test_df[!, tgt]  .= log.(test_df[!, tgt] .* 1000)
    end
    train_df[!, tgt] .= train_df[!, tgt] .* scalers[tgt]
    test_df[!, tgt]  .= test_df[!, tgt] .* scalers[tgt]
end

# mechanistic model
function SOCD_model(; SOCconc, CF, oBD, mBD)
    soct = exp.(SOCconc ./ 0.158) ./ 1000  # back to fraction
    cft = CF ./ 2.2                     # back to fraction
    BD = (oBD .* mBD) ./ (1.724f0 .* soct .* mBD .+ (1f0 .- 1.724f0 .* soct) .* oBD)
    SOCdensity = soct .* BD .* (1 .- cft)
    
    SOCdensity = log.(SOCdensity .* 1000) .* 0.165  # scale to ~[0,1]
    BD = BD .* 0.53  # scale to ~[0,1]
    return (; BD, SOCconc, CF, SOCdensity, oBD, mBD)  # supervise both BD and SOCconc
end

# param bounds
parameters = (
    SOCconc = (0.01f0, 0.0f0, 1.0f0),   # fraction
    CF      = (0.15f0, 0.0f0, 1.0f0),   # fraction,
    oBD     = (0.20f0, 0.05f0, 0.40f0),  # g/cm^3
    mBD     = (1.20f0, 0.75f0, 2.0f0),  # global
)

# define param for hybrid model
neural_param_names = [:SOCconc, :CF, :mBD]
global_param_names = [:oBD]
forcing = Symbol[]
targets = [:BD, :SOCconc, :SOCdensity, :CF]       # SOCconc is both a param and a target

# just exclude targets explicitly to be safe
predictors = Symbol.(names(train_df))[5:end-1]; # first 3 and last 1
nf = length(predictors)

# search space
batch_sizes = [32, 64, 128, 256, 512];
lrs = [1e-3, 1e-4];
acts = [swish, gelu];

# store results
results = []
best_r2 = -Inf
best_bundle = nothing

for bs in batch_sizes, lr in lrs, act in acts
    @info "Testing bs=$(bs), lr=$(lr), act=$(act)"

    hm = constructHybridModel(
        predictors,              # single NN uses a Vector of predictors
        forcing,
        targets,
        SOCD_model,
        parameters,
        neural_param_names,
        global_param_names;     
        hidden_layers = [256, 128, 64, 32, 16],
        activation    = act,
        scale_nn_outputs = true,
        input_batchnorm = true,
        start_from_default = true
    )

    # prepare data....
    (x_train, y_train) = EasyHybrid.prepare_data(hm, train_df)
    (x_val,   y_val)   = EasyHybrid.prepare_data(hm, test_df)

    res = train(
        hm, ((x_train, y_train), (x_val, y_val)), ();  
        nepochs = 200,
        batchsize = bs,
        opt = AdamW(lr),
        training_loss = :mse,
        loss_types = [:mse, :r2],
        shuffleobs = true,
        file_name = nothing,
        random_seed = 42,
        patience = 20,
        yscale = identity,
        monitor_names = [:oBD, :mBD],
        agg = mean,
        return_model = :best,
        plotting = false
    )

    # retrieve the best epoch metrics: mse and r2
    agg_name = Symbol("mean") 
    r2s  = map(vh -> getproperty(vh, agg_name),  res.val_history.r2)
    mses = map(vh -> getproperty(vh, agg_name), res.val_history.mse)
    best_idx = findmax(r2s)[2]   # index of best r2
    best_r2_here = r2s[best_idx]
    best_mse_here = mses[best_idx]

    push!(results, (bs, lr, act, best_r2_here, best_mse_here, best_idx))

    # keep the whole bundle if better
    if !isnan(best_r2_here) && best_r2_here > best_r2
        best_r2 = best_r2_here

        # map global mBD -> physical
        oBD_phys = EasyHybrid.scale_single_param(:oBD, res.ps[:oBD], hm.parameters) |> vec |> first
        oBD_raw  = res.ps[:oBD][1]  # unconstrained optimizer value

        # per-sample oBD
        mBD_phys = (hasproperty(res, :val_diffs) && hasproperty(res.val_diffs, :mBD)) ?
                collect(res.val_diffs.mBD) : nothing

        best_bundle = (
            ps = deepcopy(res.ps),
            st = deepcopy(res.st),
            model = hm,
            val_obs_pred = deepcopy(res.val_obs_pred),
            val_diffs = hasproperty(res, :val_diffs) ? deepcopy(res.val_diffs) : nothing,
            meta = (bs=bs, lr=lr, act=act, best_epoch=best_idx,
                    r2=best_r2_here, mse=best_mse_here),
            # convenience fields
            oBD_physical = oBD_phys,
            oBD_unconstr = oBD_raw,
            mBD_phys = mBD_phys
        )
    end
end

df_results = DataFrame(
    batch_size    = [r[1] for r in results],
    learning_rate = [r[2] for r in results],
    activation    = [string(r[3]) for r in results],
    r2            = [r[4] for r in results],
    mse           = [r[5] for r in results],
    best_epoch    = [r[6] for r in results]
)

out_file = joinpath(results_dir, "$(testid)_parameter_search.csv")
CSV.write(out_file, df_results)

# print best model
@assert best_bundle !== nothing "No valid model found for $testid"
bm = best_bundle
file_path = joinpath(results_dir, "$(testid)_best_model.jld2")
jldsave(file_path;
    ps=best_bundle.ps, st=best_bundle.st, model=best_bundle.model,
    val_obs_pred=best_bundle.val_obs_pred, val_diffs=best_bundle.val_diffs,
    meta=best_bundle.meta,
    mBD_phys=best_bundle.mBD_phys,
    oBD_physical=best_bundle.oBD_physical,      # use the actual field
    oBD_unconstr=best_bundle.oBD_unconstr
)

# @load joinpath(results_dir, "best_model_$(tgt).jld2") ps st model val_obs_pred meta
@info "Best for $testid: bs=$(bm.meta.bs), lr=$(bm.meta.lr), act=$(bm.meta.act), epoch=$(bm.meta.best_epoch), R2=$(round(best_r2, digits=4))"

# load predictions
jld = joinpath(results_dir, "$(testid)_best_model.jld2")
@assert isfile(jld) "Missing $(jld). Did you train & save best model for $(tname)?"
@load jld val_obs_pred meta
# split output table
val_tables = Dict{Symbol,Vector{Float64}}()
for t in targets
    # expected: t (true), t_pred (pred), and maybe :index if the framework saved it
    have_pred = Symbol(t, :_pred)
    req = Set((t, have_pred))
    @assert issubset(req, Symbol.(names(val_obs_pred))) "val_obs_pred missing $(collect(req)) for $(t). Columns: $(names(val_obs_pred))"
    val_tables[t] = val_obs_pred[:, t]./ scalers[t]
    val_tables[have_pred] = val_obs_pred[:, have_pred]./ scalers[t]
    if t in (:SOCdensity, :SOCconc)
        val_tables[Symbol("$(t)_pred")] = exp.(val_tables[Symbol("$(t)_pred")]) ./ 1000
        val_tables[t] = exp.(val_tables[t]) ./ 1000
    end
end


# helper for metrics calculation
r2_mse(y_true, y_pred) = begin
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    r2  = 1 - ss_res / ss_tot
    mse = mean((y_true .- y_pred).^2)
    (r2, mse)
end

# accuracy plots for SOCconc, BD, CF in original space
for tname in targets
    y_val_true = val_tables[tname]
    y_val_pred = val_tables[Symbol("$(tname)_pred")]

    # @assert all(in(Symbol.(names(df_out))).([tname, Symbol("$(tname)_pred")])) "Expected columns $(tname) and $(tname)_pred in saved val table."

    r2, mse = r2_mse(y_val_true, y_val_pred)

    plt = histogram2d(
        y_val_pred, y_val_true;
        nbins=(40, 40), cbar=true, xlab="Predicted", ylab="Observed",
        title = string(tname, "\nR²=", round(r2, digits=3), ", MSE=", round(mse, digits=3)),
        normalize=false
    )
    lims = extrema(vcat(y_val_true, y_val_pred))
    Plots.plot!(plt, [lims[1], lims[2]], [lims[1], lims[2]];
        color=:black, linewidth=2, label="1:1 line",
        aspect_ratio=:equal, xlims=lims, ylims=lims
    )
    savefig(plt, joinpath(results_dir, "$(testid)_accuracy_$(tname).png"))
end

# BD vs SOCconc predictions
plt = histogram2d(
    val_tables[:BD_pred], val_tables[:SOCconc_pred];
    nbins      = (30, 30),
    cbar       = true,
    xlab       = "BD",
    ylab       = "SOCconc",
    color      = cgrad(:bamako, rev=true),
    normalize  = false,
    size = (460, 400),
    xlims     = (0, 1.8),
    ylims     = (0, 0.6)
)   
savefig(plt, joinpath(results_dir, "$(testid)_BD.vs.SOCconc.png"));

# save / print parameters: mBD and per-sample oBD
# oBD global
@load jld oBD_physical
@info "Global oBD ≈ $(round(oBD_physical, digits=4))"

@load jld mBD_phys
histogram(mBD_phys; bins=:sturges, xlabel="learned mBD", ylabel="count",
          title="Distribution of learned mBD", legend=false)
vline!([mean(mBD_phys)]; lw=2, label=false)  # mean marker
@info "Saved histogram to $(joinpath(results_dir, "mBD_histogram.png"))"

# plot mBD_phys and texture
texture = CSV.read(joinpath(@__DIR__, "data/lucas_test_texture.csv"), DataFrame; normalizenames=true)
texture.mBD_phys = mBD_phys


for col in (:clay, :silt, :sand)
    p = Plots.scatter(texture[!, col], texture.BD; label="BD", xlabel=String(col), ylabel="Density", legend=:topleft)
    Plots.scatter!(texture[!, col], texture.mBD_phys; label="mBD_phys")
    title!(p, "$(col) vs density")
    display(p)
    # savefig("$(col)_vs_density.png")   # or display without saving
end

# drop rows with missings in these columns
tex = dropmissing(texture, [:clay, :sand, :BD, :mBD_phys])
cmin = 0.7 #minimum(vcat(tex.BD, tex.mBD_phys))
cmax = 1.6 #maximum(vcat(tex.BD, tex.mBD_phys))

# BD plot
p1 = Plots.scatter(tex.clay, tex.sand;
    zcolor = tex.BD,
    xlabel = "Clay",
    ylabel = "Sand",
    # colorbar_title = "BD",
    title = "observed BD",
    legend = false,
    colorbar = true,
    color = cgrad(:viridis, rev=true),
    clim = (cmin, cmax),
    markersize = 2.5, markerstrokewidth = 0,
    aspect_ratio = :equal)

# mBD_phys plot
p2 = Plots.scatter(tex.clay, tex.sand;
    zcolor = tex.mBD_phys,
    xlabel = "Clay",
    ylabel = "Sand",
    # colorbar_title = "mBD_phys",
    title = "mineral BD",
    legend = false,
    colorbar = true,
    color = cgrad(:viridis, rev=true),
    clim = (cmin, cmax),
    markersize = 2.5, markerstrokewidth = 0,
    aspect_ratio = :equal)

finalplot = Plots.plot(p1, p2, layout=(1,2), size=(900,450))
display(finalplot)
savefig(finalplot, joinpath(results_dir, "$(testid)_texture.vs.BD.png")) 

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


