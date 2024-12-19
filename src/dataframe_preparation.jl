#=
Concept of setting cells in contingency tables to 0 based on:
Ponge, J., Enbergs, M., Schüngel, M., Hellingrath, B., Karch, A., & Ludwig, S. (2021, December). 
Generating synthetic populations based on german census data. In 2021 Winter Simulation Conference (WSC) (pp. 1-12). IEEE.
=#


"""
    read_json_file(filepath::String)

Auxilary function - it returns a parsed JSON file

Arguments:
- `filepath` - path to the JSON file.
"""
function read_json_file(filepath::String)
    json = read(filepath)
    dict = JSON3.read(json)

    return dict
end


"""
    get_config_elements(config_element::JSON3.Object{Vector{UInt8}, SubArray{UInt64, 1, Vector{UInt64}, Tuple{UnitRange{Int64}}, true}})

Auxilary function - it returns a single parsed element of the config.

Arguments:
- `config_element` - single element of the config file
"""
function get_config_elements(
    config_element::JSON3.Object{
        Vector{UInt8},
        SubArray{UInt64,1,Vector{UInt64},Tuple{UnitRange{Int64}},true},
    },
)
    if_dictionary = config_element["if"]
    then_dictionary = config_element["then"]
    if_column = only(keys(if_dictionary))
    if_values = only(values(if_dictionary))
    then_column = only(keys(then_dictionary))
    then_values = only(values(then_dictionary))

    return if_column, if_values, then_column, then_values
end


"""
    get_dictionary_dfs_for_ipf(df1::DataFrame, df2::DataFrame)

Auxilary function - it returns a dictionary with data frames that are used for generation of joint distribution of attributes.

Arguments:
- `df1` - first data frame that is to be merged
- `df2` - second data frame that is to be merged
"""
function get_dictionary_dfs_for_ipf(df1::DataFrame, df2::DataFrame)
    df1[:, POPULATION_COLUMN] = Int.(round.(df1[:, POPULATION_COLUMN]))
    df2[:, POPULATION_COLUMN] = Int.(round.(df2[:, POPULATION_COLUMN]))
    
    df1_copy = select(df1, Not(POPULATION_COLUMN))
    df2_copy = select(df2, Not(POPULATION_COLUMN))
    intersecting_columns = intersect(names(df1_copy), names(df2_copy))
    
    if isempty(intersecting_columns)
        merged_attributes = crossjoin(df1_copy, df2_copy)
    else
        merged_attributes = outerjoin(df1_copy, df2_copy, on=intersecting_columns)
    end
    
    names_df1 = setdiff(names(df1_copy), names(df2_copy))
    select!(merged_attributes, vcat(names_df1, names(df2_copy)))
    #sort!(merged_attributes, reverse(names(merged_attributes)))
    
    for df in [merged_attributes, df1, df2]
        sort!(df, reverse(setdiff(names(df), [string(POPULATION_COLUMN)])))
        println("SORT COLUMNS: ", reverse(setdiff(names(df), [string(POPULATION_COLUMN)])))
    end

    dfs_for_ipf = Dict(
        "ipf_merged_attributes" => merged_attributes,
        "ipf_df1" => df1,
        "ipf_df2" => df2,
    )

    return dfs_for_ipf
end


"""
    indices_for_compute_ipf(dictionary::JSON3.Object{Vector{UInt8}, SubArray{UInt64, 1, Vector{UInt64}, Tuple{UnitRange{Int64}}, true}}, merged_attributes::DataFrame)

Auxilary function - it returns a vector of indices of rows of the data frame that meet criteria specified by single config element from the `dictionary` argument.

Arguments:
- `dictionary` - parsed element of array from config JSON file that specifies FORCED config. More information in `notebooks/config_tutorial.ipynb`.
- `merged_attributes` - data frame, whose row indices are extracted.
"""
function indices_for_compute_ipf(
    dictionary::JSON3.Object{
        Vector{UInt8},
        SubArray{UInt64,1,Vector{UInt64},Tuple{UnitRange{Int64}},true},
    },
    merged_attributes::DataFrame,
)
    temp_df = copy(merged_attributes)
    if_column, if_values, then_column, then_values = get_config_elements(dictionary)
    attribute_names = names(merged_attributes)

    if String(if_column) in attribute_names && String(then_column) in attribute_names
        #find indices for "if"s
        if typeof(if_values[1]) == String
            temp_df = filter(Symbol(if_column) => x -> in(if_values, x), temp_df)
        elseif typeof(if_values[1]) == Int
            temp_df = filter(Symbol(if_column) => x -> x in if_values, temp_df)
        end

        #find_indices for no-"then's"
        if typeof(then_values[1]) == String
            temp_df = filter(Symbol(then_column) => x -> !in(then_values, x), temp_df)
        elseif typeof(then_values[1]) == Int
            temp_df = filter(Symbol(then_column) => x -> !(x in then_values), temp_df)
        end
    end

    return temp_df[:, ID_COLUMN]
end


"""
    get_zero_indices(config_file::String, merged_attributes::DataFrame)

Auxilary function - it returns an array of all indices of rows of a data frame, that are specified by all config elements from the config JSON file.

Arguments:
- `config_file` - path of the config JSON file. More information about config file in `notebooks/config_tutorial.ipynb`.
- `merged_attributes` - data frame, whose row indices are extracted.
"""
function get_zero_indices(config_file::String, merged_attributes::DataFrame)
    config = read_json_file(config_file)
    forced_config = config["forced_config"]
    zero_indices = Int[]
    if forced_config != "missing"
        for dictionary in forced_config
            indices = indices_for_compute_ipf(dictionary, merged_attributes)
            append!(zero_indices, indices)
        end
        unique!(zero_indices)
    end
    return zero_indices
end


"""
    get_dfs_slices(dfs_for_ipf::Dict{String, DataFrame}, missing_config::JSON3.Array{JSON3.Object, Vector{UInt8}, SubArray{UInt64, 1, Vector{UInt64}, Tuple{UnitRange{Int64}}, true}})

Auxilary function - it returns a dictionary `dfs_dict` which stores dictionaries split according to the MISSING configuration.

Arguments:
- `dfs_for_ipf` - a dictionary with three key-value pairs:
    - `ipf_df1` - first data frame with distribution of population by some attributes 
    - `ipf_df2` - second data frame with distribution of population by some attributes 
    - `ipf_merged_attributes` - data frame that contains all combinations of unique values of attributes from `ipf_df1` and `ipf_df2`
- `missing_config` - configuration of type MISSING parsed from the config JSON file.
"""
function get_dfs_slices(
    dfs_for_ipf::Dict{String,DataFrame},
    missing_config::JSON3.Array{
        JSON3.Object,
        Vector{UInt8},
        SubArray{UInt64,1,Vector{UInt64},Tuple{UnitRange{Int64}},true},
    },
)
    df1 = copy(dfs_for_ipf["ipf_df1"])
    df2 = copy(dfs_for_ipf["ipf_df2"])
    ipf_merged_attributes = copy(dfs_for_ipf["ipf_merged_attributes"]) 

    #filtering the ipf_merged_attributes DataFrame
    missing_dfs_array = DataFrame[]
    for dictionary in missing_config
        if_column, if_values, then_column, then_values = get_config_elements(dictionary)

        if String(if_column) in names(ipf_merged_attributes) &&
           String(then_column) in names(ipf_merged_attributes)
            ipf_merged_attributes_missing = copy(dfs_for_ipf["ipf_merged_attributes"])

            if typeof(if_values[1]) == String
                ipf_merged_attributes = filter(
                    Symbol(if_column) => x -> !in(if_values, x),
                    ipf_merged_attributes,
                )
                ipf_merged_attributes_missing = filter(
                    Symbol(if_column) => x -> in(if_values, x),
                    ipf_merged_attributes_missing,
                )
            elseif typeof(if_values[1]) == Int
                ipf_merged_attributes = filter(
                    Symbol(if_column) => x -> !(x in if_values),
                    ipf_merged_attributes,
                )
                ipf_merged_attributes_missing = filter(
                    Symbol(if_column) => x -> x in if_values,
                    ipf_merged_attributes_missing,
                )
            end
            push!(missing_dfs_array, ipf_merged_attributes_missing)
        end

    end

    #filtering the df1 and df2 DataFrame
    missing_dfs1_array = DataFrame[]
    missing_dfs2_array = DataFrame[]
    for dictionary in missing_config
        if_column, if_values, then_column, then_values = get_config_elements(dictionary)

        if String(if_column) in names(ipf_merged_attributes) &&
           String(then_column) in names(ipf_merged_attributes)
            df1_missing = copy(dfs_for_ipf["ipf_df1"])
            df2_missing = copy(dfs_for_ipf["ipf_df2"])

            if typeof(if_values[1]) == String
                if if_column in names(df1)
                    df1 = filter(Symbol(if_column) => x -> !in(if_values, x), df1)
                    df1_missing =
                        filter(Symbol(if_column) => x -> in(if_values, x), df1_missing)
                end
                if if_column in names(df2)
                    df2 = filter(Symbol(if_column) => x -> !in(if_values, x), df2)
                    df2_missing =
                        filter(Symbol(if_column) => x -> in(if_values, x), df2_missing)
                end
            elseif typeof(if_values[1]) == Int
                if String(if_column) in names(df1)
                    df1 = filter(Symbol(if_column) => x -> !(x in if_values), df1)
                    df1_missing =
                        filter(Symbol(if_column) => x -> x in if_values, df1_missing)
                end
                if String(if_column) in names(df2)
                    df2 = filter(Symbol(if_column) => x -> !(x in if_values), df2)
                    df2_missing =
                        filter(Symbol(if_column) => x -> x in if_values, df2_missing)
                end
            end
            push!(missing_dfs1_array, df1_missing)
            push!(missing_dfs2_array, df2_missing)
        end

    end

    ipf_merged_attributes_missing = unique(vcat(missing_dfs_array...))
    df1_missing = unique(vcat(missing_dfs1_array...))
    df2_missing = unique(vcat(missing_dfs2_array...))

    dfs_dict = Dict{String,Dict{String,DataFrame}}(
        "dfs_for_ipf" => Dict(
            "ipf_df1" => df1,
            "ipf_df2" => df2,
            "ipf_merged_attributes" => ipf_merged_attributes,
        )
    )
    if !any(df -> isempty(df), [df1_missing, df2_missing, ipf_merged_attributes_missing])
        dfs_dict["dfs_missing_config"] = Dict(
            "ipf_df1" => df1_missing,
            "ipf_df2" => df2_missing,
            "ipf_merged_attributes" => ipf_merged_attributes_missing,
        )
    end

    return dfs_dict
end


"""
    filter_dfs_for_ipf_by_missing_config(dfs_for_ipf::Dict{String, DataFrame}, config_file::Union{String, Nothing})

Auxilary function - it returns a dictionary `dfs_dict` with dictionary keys `"dfs_for_ipf"` and, optionally if MISSING config is defined, `"dfs_missing_config"`.

Arguments:
- `dfs_for_ipf` - a dictionary with three key-value pairs:
    - `ipf_df1` - first data frame with distribution of population by some attributes 
    - `ipf_df2` - second data frame with distribution of population by some attributes 
    - `ipf_merged_attributes` - data frame that contains all combinations of unique values of attributes from `ipf_df1` and `ipf_df2`
- `missing_config` - path to config JSON file
"""
function filter_dfs_for_ipf_by_missing_config(
    dfs_for_ipf::Dict{String,DataFrame},
    config_file::Union{String,Nothing},
)
    dfs_dict = Dict{String,Dict{String,DataFrame}}()
    if config_file === nothing
        dfs_dict["dfs_for_ipf"] = dfs_for_ipf
    else
        config_file = read_json_file(config_file)
        missing_config = config_file["missing_config"]
        if missing_config != "missing"
            dfs_dict = get_dfs_slices(dfs_for_ipf, missing_config)
        else
            dfs_dict["dfs_for_ipf"] = dfs_for_ipf
        end
    end
    return dfs_dict
end


"""
    merge_attributes(df1::DataFrame, df2::DataFrame; config_file::Union{String, Nothing})

Auxilary function - it returns a dictionary with all the data frames that are needed in order to generate a joint distribution of attributes of two data frames.

Arguments:
- `df1` - the first data frame, that is to be merged
- `df2` - the second data frame, that is to be merged
- `config_file` - optional argument; path to the JSON file that specifies merging config
"""
function merge_attributes(
    df1::DataFrame,
    df2::DataFrame;
    config_file::Union{String,Nothing},
)
    #merge the dfs with merginal attributes into 1 dataframe
    dfs_for_ipf = get_dictionary_dfs_for_ipf(df1, df2)
    dfs_dict = filter_dfs_for_ipf_by_missing_config(dfs_for_ipf, config_file)

    #add :compute_ipf column. 0 -> do not compute ipf; 1 -> compute ipf
    if haskey(dfs_dict, "dfs_missing_config")
        merged_attributes = copy(dfs_dict["dfs_missing_config"]["ipf_merged_attributes"])
        merged_attributes.:compute_ipf = Int.(ones(nrow(merged_attributes)))
        dfs_dict["dfs_missing_config"]["ipf_merged_attributes"] = merged_attributes
    end

    merged_attributes = copy(dfs_dict["dfs_for_ipf"]["ipf_merged_attributes"])
    
    if config_file === nothing
        merged_attributes.:compute_ipf = Int.(ones(nrow(merged_attributes)))
    else
        #add id column
        merged_attributes[:, ID_COLUMN] = collect(1:nrow(merged_attributes))
        #indices set to 0
        zero_indices = get_zero_indices(config_file, merged_attributes)
        #create compute_ipf column
        merged_attributes.:compute_ipf =
            [(i in zero_indices) ? 0 : 1 for i = 1:nrow(merged_attributes)]
        #drop id column
        merged_attributes = merged_attributes[:, Not(ID_COLUMN)]
    end

    dfs_dict["dfs_for_ipf"]["ipf_merged_attributes"] = merged_attributes



    return dfs_dict
end
