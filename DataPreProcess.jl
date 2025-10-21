# CC BY-SA 4.0
using Revise
using EasyHybrid
using EasyHybrid.MLUtils
using Plots

# ? move the `csv` file into the `BulkDSOC/data` folder (create folder)
df_o = CSV.read(joinpath(@__DIR__, "./data/lucas_overlaid.csv"), DataFrame, normalizenames=true)
println(size(df_o));

## target: BD g/cm3, SOCconc g/kg, CF [0,1]
target_names = [:BD, :SOCconc, :CF, :SOCdensity];
rename!(df_o, :bulk_density_fe => :BD, :soc => :SOCconc, :coarse_vol => :CF); # rename as in hybrid model
df_o[!, :SOCconc] .= df_o[!, :SOCconc] ./ 1000; # convert to fraction, [0,1]
df_o[!,:SOCdensity] = df_o.BD .* df_o.SOCconc .* (1 .- df_o.CF); # SOCdensity g/cm3

coords = collect(zip(df_o.lat, df_o.lon))

# t clean covariates
names_cov = Symbol.(names(df_o))[19:end-1]
# names_meta = Symbol.(names(df_o))[1:18]

# Fix soilsuite and cropland extent columns
for col in names_cov
    if occursin("_soilsuite_", String(col))
        df_o[!, col] = replace(df_o[!, col], missing => 0)
    elseif occursin("cropland_extent_", String(col))
        df_o[!, col] = replace(df_o[!, col], missing => 0)
        df_o[!, col] .= ifelse.(df_o[!, col] .> 0, 1, 0)
    end
end

# rm missing values: 1. >5%, drop col; 2. <=5%, drop row
cols_to_drop_row = Symbol[]
cols_to_drop_col = Symbol[] 
for col in names_cov
    n_missing = count(ismissing, df_o[!, col])
    frac_missing = n_missing / nrow(df_o)
    if frac_missing > 0.05
        println(n_missing, " ", col)
        select!(df_o, Not(col))  # drop the column
        push!(cols_to_drop_col, col)  
    elseif n_missing > 0
        # println(n_missing, " ", col)
        push!(cols_to_drop_row, col)  # collect column name
    end

    if occursin("CHELSA_kg", String(col)) 
        push!(cols_to_drop_col, col) 
        select!(df_o, Not(col))  # rm kg catagorical col
    end 
end

names_cov = filter(x -> !(x in cols_to_drop_col), names_cov) # remove cols-to-drop from names_cov
if !isempty(cols_to_drop_row) 
    df_o = subset(df_o, cols_to_drop_row .=> ByRow(!ismissing)) # drop rows with missing values in cols_to_drop_row
end
println(size(df_o))

cols_to_drop_col = Symbol[] 
for col in names_cov
    if std(df_o[:,col])==0
        push!(cols_to_drop_col, col)  # rm constant col (std==0)
        select!(df_o, Not(col))
    end
end
names_cov = filter(x -> !(x in cols_to_drop_col), names_cov) # remove cols-to-drop from names_cov
println(size(df_o))


# for col in names_cov # to check covairate distribution
#     println(string(col)[1:10], ' ', round(std(df[:, col]); digits=2), ' ', round(mean(df[:, col]); digits=2))
# end

# # Normalize covariates with std>1
means = mean.(eachcol(df_o[:, names_cov]))
stds = std.(eachcol(df_o[:, names_cov]))
for col in names_cov
    df_o[!, col] = Float64.(df_o[!, col])
end
df_o[:, names_cov] .= (df_o[:, names_cov] .- means') ./ stds'

println(size(df))
df_o.row_id = 1:nrow(df_o);
# CSV.write(joinpath(@__DIR__, "data/lucas_preprocessed.csv"), df)

# fillna texture
g = groupby(df_o, :id)
for col in [:clay, :silt, :sand]
    transform!(g, col => (x -> begin
        i = findfirst(!ismissing, x)          # index of first non-missing in this id
        i === nothing ? x : coalesce.(x, x[i]) # fill missings with that value
    end) => col)
end


# split train and test
raw_val = dropmissing(df_o, target_names);
Random.seed!(42);
eligible = raw_val.row_id;
train_ids, val_ids = splitobs(collect(eligible); at=0.8, shuffle=true);
test  = df_o[in.(df_o.row_id, Ref(val_ids)), [:time, :BD, :SOCconc, :CF, :SOCdensity, names_cov...]]
test_texture = df_o[in.(df_o.row_id, Ref(val_ids)), [:time, :lat, :lon, :id, :clay, :silt, :sand, :BD]]
train = df_o[.!in.(df_o.row_id, Ref(val_ids)), [:time, :BD, :SOCconc, :CF, :SOCdensity, names_cov...]]

# CSV.write(joinpath(@__DIR__, "data/lucas_train.csv"), train)
# CSV.write(joinpath(@__DIR__, "data/lucas_test.csv"), test)
CSV.write(joinpath(@__DIR__, "data/lucas_test_texture.csv"), test_texture)

# # plot BD vs SOCconc
# bd_lims = extrema(skipmissing(df[:, "BD"]))      
# soc_lims = extrema(skipmissing(df[:, "SOCconc"]))
# plt = histogram2d(
#     df[:, "BD"], df[:, "SOCconc"];
#     nbins      = (30, 30),
#     cbar       = true,
#     xlab       = "BD",
#     ylab       = "SOCconc",
#     xlims=bd_lims, ylims=soc_lims,
#     #title      = "SOCdensity-MTD\nR2=$(round(r2, digits=3)), MAE=$(round(mae, digits=3)), bias=$(round(bias, digits=3))",
#     color      = cgrad(:bamako, rev=true),
#     normalize  = false,
#     size = (460, 400),
#     xlims     = (0, 1.8),
#     ylims     = (0, 0.6)
# )   
# savefig(plt, joinpath(@__DIR__, "./eval/00_truth_BD.vs.SOCconc.png"))

# # check distribution of BD, SOCconc, CF
# for col in ["BD", "SOCconc", "CF", "SOCdensity"]
#     # values = log10.(df[:, col])
#     values = df_o[:, col]
#     histogram(
#         values;
#         bins = 50,
#         xlabel = col,
#         ylabel = "Frequency",
#         title = "Histogram of $col",
#         lw = 1,
#         legend = false
#     )
#     savefig(joinpath(@__DIR__, "./eval/histogram_$col.png"))
# end