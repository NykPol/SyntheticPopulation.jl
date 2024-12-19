#=
Algorithm for merging two data frames is inspired by:
Guo, J. Y., & Bhat, C. R. (2007). Population synthesis for microsimulating travel behavior. 
Transportation Research Record, 2014(1), 92-101.
=#


"""
    fit_ipf(dfs_for_ipf::Dict{String, DataFrame}; ipf_population::String)

Auxilary function - it returns an estimated joint distribution of two data frames without common attributes by applying an IPF procedure.

Arguments:
- `dfs_for_ipf` - a dictionary with three key-value pairs:
    - `ipf_df1` - first data frame with distribution of population by some attributes 
    - `ipf_df2` - second data frame with distribution of population by some attributes 
    - `ipf_merged_attributes` - data frame that contains all combinations of unique values of attributes from `ipf_df1` and `ipf_df2`
- `ipf_population` - argument defining what will be the total population used in the computation performed by IPF algoriithm
"""
function fit_ipf(dfs_for_ipf::Dict{String,DataFrame}; ipf_population::String)
    ipf_merged_attributes = dfs_for_ipf["ipf_merged_attributes"]
    ipf_df1 = dfs_for_ipf["ipf_df1"]
    ipf_df2 = dfs_for_ipf["ipf_df2"]

    println("ipf_merged_attributes: ", ipf_merged_attributes)
    println("ipf_df1: ", ipf_df1)
    println("ipf_df2: ", ipf_df2)

    #initialize sample matrix as seed for IPF
    input_sample =
        reshape(ipf_merged_attributes.:compute_ipf, (nrow(ipf_df1), nrow(ipf_df2)))
    println("input_sample: ", input_sample)
    #initiate input marginal distributions for IPF
    input_marginals = vcat([ipf_df1[:, POPULATION_COLUMN]], [ipf_df2[:, POPULATION_COLUMN]])
    population_size_difference =
        abs(sum(ipf_df1[:, POPULATION_COLUMN]) - sum(ipf_df2[:, POPULATION_COLUMN]))

    #fit using the iterative proportional fitting
    if population_size_difference == 0
        fac = ipf(input_sample, input_marginals)
        Z = Array(fac) .* input_sample
        if isnan(Z[1]) #error handling
            input_sample = input_sample .+ 1e-20
            fac = ipf(input_sample, input_marginals)
            Z = Array(fac) .* input_sample
        end
        Z = Int.(round.(Z))
    else
        af = ipf(input_sample, ArrayMargins(input_marginals))
        if isnan(Array(af)[1]) #error handling
            input_sample = input_sample .+ 1e-20 # to avoid dividing by zero
            af = ipf(input_sample, input_marginals)
        end
        X_prop = input_sample ./ sum(input_sample)
        Z = X_prop .* Array(af)
        if ipf_population == "max"
            population_size =
                max(sum(ipf_df1[:, POPULATION_COLUMN]), sum(ipf_df2[:, POPULATION_COLUMN]))
        elseif ipf_population == "min"
            population_size =
                min(sum(ipf_df1[:, POPULATION_COLUMN]), sum(ipf_df2[:, POPULATION_COLUMN]))
        end
        Z = population_size .* Z
        Z = Int.(round.(Z)) #error alert: if the numbers are low, some numbers can be rounded to 0. e.g. total will be 10 instead of 15
    end
    println("Z: ", Z)
    #generate output df from ipf
    joint_distribution = copy(ipf_merged_attributes)
    println("joint_distribution: ", joint_distribution)
    joint_distribution[:, POPULATION_COLUMN] = vec(Z)
    println("vec(Z): ", vec(Z))
    joint_distribution = joint_distribution[:, Not(:compute_ipf)]

    return joint_distribution
end


"""
    get_dfs_for_ipf_slice(dfs_for_ipf::Dict{String, DataFrame}, unique_value::Any, column::Union{String, Symbol})

Auxilary function - it returns filtered data frames given filtering config. 

Arguments:
- `dfs_for_ipf` - a dictionary with three key-value pairs:
    - `ipf_df1` - first data frame with distribution of population by some attributes 
    - `ipf_df2` - second data frame with distribution of population by some attributes 
    - `ipf_merged_attributes` - data frame that contains all combinations of unique values of attributes from `ipf_df1` and `ipf_df2`
- `column` - a column of the data frames, on which the filtering is done
- `unique_value` - a value, for which the filtering is done
"""
function get_dfs_for_ipf_slice(
    dfs_for_ipf::Dict{String,DataFrame},
    unique_value::Any,
    column::Union{String,Symbol},
)
    dfs_for_ipf_slice = Dict{String,DataFrame}()
    ipf_merged_attributes = copy(dfs_for_ipf["ipf_merged_attributes"])
    ipf_df1 = copy(dfs_for_ipf["ipf_df1"])
    ipf_df2 = copy(dfs_for_ipf["ipf_df2"])

    ipf_df1_slice = filter(column => n -> n == unique_value, ipf_df1)
    ipf_df2_slice = filter(column => n -> n == unique_value, ipf_df2)
    ipf_merged_attributes_slice =
        filter(column => n -> n == unique_value, ipf_merged_attributes)

    dfs_for_ipf_slice["ipf_df1"] = ipf_df1_slice
    dfs_for_ipf_slice["ipf_df2"] = ipf_df2_slice
    dfs_for_ipf_slice["ipf_merged_attributes"] = ipf_merged_attributes_slice

    return dfs_for_ipf_slice
end


"""
    compute_joint_distributions(dfs_for_ipf::Dict{String, DataFrame}; ipf_population::String, shared_columns::Vector{String} = String[])

Auxilary function - it returns an estimated joint distribution of two data frames.

Arguments:
- `dfs_for_ipf` - a dictionary with three key-value pairs:
    - `ipf_df1` - first data frame with distribution of population by some attributes 
    - `ipf_df2` - second data frame with distribution of population by some attributes 
    - `ipf_merged_attributes` - data frame that contains all combinations of unique values of attributes from `ipf_df1` and `ipf_df2`
- `ipf_population` - argument defining what will be the total population used for computation in the IPF algorithm that is called within this function.
- `shared_columns` - an optional argument used when the function is called recursively.
"""
function compute_joint_distributions(
    dfs_for_ipf::Dict{String,DataFrame};
    ipf_population::String,
    shared_columns::Vector{String} = String[],
)
    ipf_df1 = copy(dfs_for_ipf["ipf_df1"])
    ipf_df2 = copy(dfs_for_ipf["ipf_df2"])
    ipf_merged_attributes = copy(dfs_for_ipf["ipf_merged_attributes"])

    # for df in [ipf_df1, ipf_df2, ipf_merged_attributes]
    #     sort!(df, reverse(setdiff(names(df), [string(POPULATION_COLUMN)])))
    #     println("SORT COLUMNS: ", reverse(setdiff(names(df), [string(POPULATION_COLUMN)])))
    # end

    if nrow(ipf_df1) == 0 && nrow(ipf_df2) == nrow(ipf_merged_attributes)
        ipf_merged_attributes[:, POPULATION_COLUMN] = ipf_df2[:, POPULATION_COLUMN]
        return ipf_merged_attributes

    elseif nrow(ipf_df2) == 0 && nrow(ipf_df1) == nrow(ipf_merged_attributes)
        ipf_merged_attributes[:, POPULATION_COLUMN] = ipf_df1[:, POPULATION_COLUMN]
        return ipf_merged_attributes

    else
        intersecting_columns = intersect(names(ipf_df1), names(ipf_df2))
        deleteat!(
            intersecting_columns,
            findall(x -> x == string(POPULATION_COLUMN), intersecting_columns),
        )
        intersecting_columns = setdiff(intersecting_columns, shared_columns)

        if length(intersecting_columns) == 0
            return fit_ipf(dfs_for_ipf, ipf_population = ipf_population)
        else
            attributes = copy(shared_columns)
            push!(attributes, intersecting_columns[1])
            unique_values_of_attribute = unique(ipf_df1[:, intersecting_columns[1]])

            if length(unique_values_of_attribute) == 1
                return fit_ipf(dfs_for_ipf, ipf_population = ipf_population)
            end

            df_list = DataFrame[]
            for unique_value in unique_values_of_attribute
                dfs_for_ipf_slice = copy(dfs_for_ipf)
                dfs_for_ipf_slice = get_dfs_for_ipf_slice(
                    dfs_for_ipf_slice,
                    unique_value,
                    intersecting_columns[1],
                )
                joint_distribution = compute_joint_distributions(
                    dfs_for_ipf_slice;
                    ipf_population = ipf_population,
                    shared_columns = attributes,
                )

                push!(df_list, joint_distribution)
            end

            return reduce(vcat, df_list)
        end
    end
end


"""
    apply_missing_config(joint_distribution::DataFrame, missing_config::JSON3.Array{JSON3.Object, Vector{UInt8}, SubArray{UInt64, 1, Vector{UInt64}, Tuple{UnitRange{Int64}}, true}})

Auxilary function - it applies the configuration of type MISSING that is specified in the config JSON file. More information on how the config works is specified in `notebooks/config_tutorial.ipynb`.

Arguments:
- `joint_distribution` - a data frame with aggregated generated population.
- `missing_config` - configuration of type MISSING parsed from the config JSON file.
"""
function apply_missing_config(
    joint_distribution::DataFrame,
    missing_config::JSON3.Array{
        JSON3.Object,
        Vector{UInt8},
        SubArray{UInt64,1,Vector{UInt64},Tuple{UnitRange{Int64}},true},
    },
)
    joint_distribution[:, ID_COLUMN] = collect(1:nrow(joint_distribution))
    attribute_names = names(joint_distribution)
    for dictionary in missing_config
        if_column, if_values, then_column, then_values = get_config_elements(dictionary)

        if String(if_column) in attribute_names && String(then_column) in attribute_names
            temp_df = copy(joint_distribution)
            allowmissing!(joint_distribution)

            #find indices for "if"s
            if typeof(if_values[1]) == String
                temp_df = filter(Symbol(if_column) => x -> in(if_values, x), temp_df)
            elseif typeof(if_values[1]) == Int
                temp_df = filter(Symbol(if_column) => x -> x in if_values, temp_df)
            end
            joint_distribution[temp_df[:, ID_COLUMN], then_column] .= missing
        end
    end

    #split-apply-combine procedure for the columns including missing
    deleteat!(
        attribute_names,
        findall(
            x -> String(x) in [string(POPULATION_COLUMN), string(ID_COLUMN)],
            attribute_names,
        ),
    )
    aggregated_joint_distribution = groupby(joint_distribution, attribute_names)
    aggregated_joint_distribution =
        combine(aggregated_joint_distribution, POPULATION_COLUMN => sum)

    #return proper dataframes
    aggregated_joint_distribution[:, POPULATION_COLUMN] =
        convert.(
            Int,
            aggregated_joint_distribution[:, Symbol(string(POPULATION_COLUMN) * "_sum")],
        )
    aggregated_joint_distribution =
        aggregated_joint_distribution[:, Not(Symbol(string(POPULATION_COLUMN) * "_sum"))]

    return aggregated_joint_distribution
end


"""
    generate_joint_distribution(marginal_distributions::DataFrame ...; config_file::Union{Nothing, String} = nothing)

Main function - it returns a data frame with distribution of population with respect to all attributes from input data frames.

Arguments:
- `marginal_distributions` - an array of data frames. Each data frame contains distribution of population with respect to given attributes. Detailed description of the format in `notebooks/dataframe_formats.ipynb`.
- `config_file` - optional argument; config JSON file specifying the configuration of the algorithm. Usage guide in `notebooks/config_tutorial.ipynb`.
"""
function generate_joint_distribution(
    marginal_distributions::DataFrame...;
    config_file::Union{Nothing,String} = nothing,
)
    if length(marginal_distributions) == 1
        joint_distribution = marginal_distributions[1]
        joint_distribution[:, POPULATION_COLUMN] =
            Int.(round.(joint_distribution[:, POPULATION_COLUMN]))
        joint_distribution[:, ID_COLUMN] = collect(1:nrow(joint_distribution))
    else
        joint_distribution = marginal_distributions[1]
        for i = 2:(length(marginal_distributions))
            dfs_dict = merge_attributes(
                joint_distribution,
                marginal_distributions[i];
                config_file = config_file,
            )
            if haskey(dfs_dict, "dfs_missing_config")
                joint_distribution = compute_joint_distributions(
                    dfs_dict["dfs_for_ipf"],
                    ipf_population = "min",
                )
                joint_distribution_missing_config = compute_joint_distributions(
                    dfs_dict["dfs_missing_config"],
                    ipf_population = "min",
                )
                
                joint_distribution =
                    vcat(joint_distribution, joint_distribution_missing_config)
                
            else
                joint_distribution = compute_joint_distributions(
                    dfs_dict["dfs_for_ipf"],
                    ipf_population = "max",
                )
                
            end
        end
    end

    #set the attribute values from MISSING config file to missing
    if config_file !== nothing
        config_file = read_json_file(config_file)
        missing_config = config_file["missing_config"]
        if missing_config != "missing"
            aggregated_joint_distribution =
                apply_missing_config(joint_distribution, missing_config)
        else
            aggregated_joint_distribution = joint_distribution
        end
    else
        aggregated_joint_distribution = joint_distribution
    end

    #add ID column for
    aggregated_joint_distribution[:, ID_COLUMN] =
        collect(1:nrow(aggregated_joint_distribution))
    aggregated_joint_distribution = select(aggregated_joint_distribution, ID_COLUMN, :)

    return aggregated_joint_distribution
end
