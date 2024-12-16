using JuMP
using GLPK
using HiGHS
using Cbc
using DataFrames
using ShiftedArrays
using LinearAlgebra

function findrow(cumulative_population, individual_id)
    for i = 1:length(cumulative_population)
        if individual_id <= cumulative_population[i]
            return i
        end
    end
    return 0  # If individual_id is below the last index, return the last index
end



"""
    add_indices_range_to_indiv(aggregated_individuals::DataFrame)

Adds individual index ranges to the `aggregated_individuals` DataFrame based on population counts, calculating a range of individual indices for each row and storing it in a new column, `indiv_indices`.

# Arguments
- `aggregated_individuals::DataFrame`: A DataFrame containing individual data with at least a `population` column that represents the count of individuals for each row.

# Returns
- `DataFrame`: The modified `aggregated_individuals` DataFrame with a new column, `indiv_indices`, representing the individual index range for each row.
"""
function add_indices_range_to_indiv(aggregated_individuals::DataFrame)
    aggregated_individuals = copy(aggregated_individuals)
    aggregated_individuals[!,"indiv_range_to"] .= cumsum(aggregated_individuals[!, POPULATION_COLUMN])
    aggregated_individuals[!, "indiv_range_from"] = replace(ShiftedArrays.lag(aggregated_individuals[!, "indiv_range_to"], 1), missing => 0)
    aggregated_individuals[!, "indiv_indices"] = map(row -> row.indiv_range_from+1:row.indiv_range_to, eachrow(aggregated_individuals))
    rename!(aggregated_individuals,:indiv_range_to => :cum_population)
    aggregated_individuals = select(aggregated_individuals, Not([:indiv_range_from]))
    return aggregated_individuals
end



"""
    add_individual_flags(aggregated_individuals::DataFrame)

Adds useful categorical flags to the `aggregated_individuals` DataFrame, determining whether each individual is an adult, a potential parent, or a potential child based on age and marital status.

# Arguments
- `aggregated_individuals::DataFrame`: A DataFrame containing individual data.

# Returns
- `DataFrame`: The modified `aggregated_individuals` DataFrame.
"""
function add_individual_flags(aggregated_individuals::DataFrame)
    aggregated_individuals = copy(aggregated_individuals)
    aggregated_individuals[!, "is_adult"] = float.(aggregated_individuals[!, AGE_COLUMN] .>= MINIMUM_ADULT_AGE)
    aggregated_individuals[!, "is_potential_child"] = float.((aggregated_individuals[!, "is_adult"] .== false) .|| (aggregated_individuals[!, MARITALSTATUS_COLUMN] .!= AVAILABLE_FOR_MARRIAGE) .&& aggregated_individuals[!, AGE_COLUMN] .< MINIMUM_ADULT_AGE + 40)
    aggregated_individuals[!, "is_married_male"] = float.((coalesce.(aggregated_individuals[!, MARITALSTATUS_COLUMN], "") .== AVAILABLE_FOR_MARRIAGE) .&& (aggregated_individuals[!, SEX_COLUMN] .== 'M'))
    aggregated_individuals[!, "is_married_female"] = float.((coalesce.(aggregated_individuals[!, MARITALSTATUS_COLUMN], "") .== AVAILABLE_FOR_MARRIAGE) .&& (aggregated_individuals[!, SEX_COLUMN] .== 'F'))
    aggregated_individuals[!, "is_potential_parent"] = float.((aggregated_individuals[!, "is_married_male"] .== true) .|| (aggregated_individuals[!, "is_married_female"] .== true))
    return aggregated_individuals
end



"""
    add_indices_range_to_hh(aggregated_households::DataFrame)

Adds household index ranges to the `aggregated_households` DataFrame based on population counts, creating a new column `hh_indices` that assigns a unique range of indices to each household based on cumulative population values.

# Arguments
- `aggregated_households::DataFrame`: A DataFrame containing household data with at least a `population` column representing the number of individuals in each household.

# Returns
- `DataFrame`: The modified `aggregated_households` DataFrame with the new column `hh_indices`, which lists index ranges for each household.
"""
function add_indices_range_to_hh(aggregated_households::DataFrame)
    aggregated_households = copy(aggregated_households)
    aggregated_households[!,"hh_range_to"] .= cumsum(aggregated_households[!, POPULATION_COLUMN])
    aggregated_households[!, "hh_range_from"] = replace(ShiftedArrays.lag(aggregated_households[!, "hh_range_to"], 1), missing => 0)
    aggregated_households[!, "hh_indices"] = map(row -> row.hh_range_from+1:row.hh_range_to, eachrow(aggregated_households))
    rename!(aggregated_households,:hh_range_to => :cum_population)
    aggregated_households = select(aggregated_households, Not([:hh_range_from]))
    return aggregated_households
end

"""
    disaggr_ipf_individuals(aggregated_individuals::DataFrame) -> DataFrame

Disaggregates a DataFrame of aggregated individuals into individual records based on the population count for each group.

### Arguments
- `aggregated_individuals::DataFrame`: A DataFrame containing aggregated data for different groups. The DataFrame must have at least a `population` column indicating the number of individuals in each group.

### Returns
- A DataFrame where each row corresponds to an individual.

### Details
- Groups with a population of zero or missing are skipped.
- The `id` column will be generated as a new sequential identifier for each individual.
- The function will ensure that the column types across the disaggregated DataFrame are consistent using `promote=true` when appending new rows.
"""

function disaggr_ipf_individuals(aggregated_individuals::DataFrame)
    # Create an empty DataFrame to store disaggregated individuals
    disaggregated_individuals = DataFrame()

    # Loop over each row in aggregated_individuals
    for row in eachrow(aggregated_individuals)
        # Get the number of individuals in this group
        pop = row[:population]
        
        # Skip rows with missing or zero population
        if isnothing(pop) || pop == 0
            continue
        end
        
        # Create expanded rows for the current group
        expanded_rows = DataFrame(NamedTuple(
            (Symbol(col) => repeat([row[col]], pop) for col in names(aggregated_individuals))
        ))
        
        # Ensure column types match across DataFrames (use promote=true)
        append!(disaggregated_individuals, expanded_rows; promote=true)
    end

    rename!(disaggregated_individuals, :id => :agg_ind_id)
    disaggregated_individuals[!, "id"] .= 1:nrow(disaggregated_individuals)
    select!(disaggregated_individuals, Not([:population, :cum_population, :indiv_indices]))   

    return disaggregated_individuals
end

"""
    define_and_run_optimization(aggregated_individuals::DataFrame,
                                 aggregated_households::DataFrame,
                                 hh_size1_indices::Vector{Int},
                                 hh_size2_indices::Vector{Int},
                                 hh_size3plus_indices::Vector{Int},
                                 hh_capacity::Vector{Int},
                                 adult_indices::Vector{Int},
                                 married_male_indices::Vector{Int},
                                 married_female_indices::Vector{Int},
                                 parent_indices::Vector{Int},
                                 child_indices::Vector{Int})

Run an optimization linear programming model to allocate individuals to households based on specific constraints related to household size and family structure.

# Arguments
- `aggregated_individuals::DataFrame`: A DataFrame containing the population data for individuals, including identifiers and demographic details.
- `aggregated_households::DataFrame`: A DataFrame containing the population data for households, including identifiers and household size information.
- `hh_size1_indices::Vector{Int}`: Indices of households that can accommodate 1 individual.
- `hh_size2_indices::Vector{Int}`: Indices of households that can accommodate 2 individuals.
- `hh_size3plus_indices::Vector{Int}`: Indices of households that can accommodate 3 or more individuals.
- `hh_capacity::Vector{Int}`: A vector containing the capacity of households.
- `adult_indices::Vector{Int}`: Indices of individuals classified as adults.
- `married_male_indices::Vector{Int}`: Indices of married male individuals.
- `married_female_indices::Vector{Int}`: Indices of married female individuals.
- `child_indices::Vector{Int}`: Indices of individuals classified as children.

# Returns
- `Matrix{Float64}`: A vector containing the allocation results, where each element indicates the household assigned to each individual. If an individual is not allocated to any household, the entry will be `missing`.
"""
function define_and_run_optimization(disaggregated_ipf_individuals::DataFrame, hh_capacity::Vector{Int})
    # Optimization
    # Define data structures for optimization

    # Total n
    n_hh = Int(length(hh_capacity))
    n_indiv = Int(nrow(disaggregated_ipf_individuals))
    n_potential_children = Int(sum(disaggregated_ipf_individuals[!,"is_potential_child"]))
    println("N households: ", n_hh)
    println("N invididuals: ", n_indiv)
    println("N potential children: ", n_potential_children)
    flush(stdout)

    # Vectors
    is_potential_child = Vector{Float64}(disaggregated_ipf_individuals[!,"is_potential_child"])
    is_married_male = Vector{Float64}(disaggregated_ipf_individuals[!,"is_married_male"])
    is_married_female = Vector{Float64}(disaggregated_ipf_individuals[!,"is_married_female"])
    is_potential_parent = Vector{Float64}(disaggregated_ipf_individuals[!,"is_potential_parent"])
    is_adult = Vector{Float64}(disaggregated_ipf_individuals[!,"is_adult"])
    age_vector = Vector{Float64}(float.(disaggregated_ipf_individuals[!,"age"]))
    println("Vectors are defined")
    flush(stdout)

    # Create a new optimization mode
    model = Model(HiGHS.Optimizer)
    set_attribute(model, "mip_rel_gap", 0.01)
    set_attribute(model, "mip_heuristic_effort", 0.25)
    println("Optimization model is defined")
    flush(stdout)

    # Define decision variables: a binary allocation matrix where
    # allocation[i, j] indicates whether individual i is assigned to household j
    @variable(model, allocation[1:n_indiv, 1:n_hh], Bin, start = 0)
    @variable(model, 0 <= household_inhabited[1:n_hh]  <= 1, start = 0 )
    @variable(model, 0 <= household_married_male_inhabited[1:n_hh]  <= 1, start = 0 )
    @variable(model, 0 <= household_married_female_inhabited[1:n_hh]  <= 1, start = 0 )
    @variable(model, 0 <= household_children_inhabited[1:n_hh]  <= 1, start = 0 )
    #@variable(model, male_parent_relaxation[1:n_potential_children, 1:n_hh], Bin, start = 0)
    #@variable(model, female_parent_relaxation[1:n_potential_children, 1:n_hh], Bin, start = 0)
    #@variable(model, penalty[1:n_hh], Int, lower_bound=0, start = 0)
    println("Decision variables are defined")
    flush(stdout)


    # Define the objective function: maximize the total number of assigned individuals
    @objective(model, Max, sum(allocation))
    println("Objective function is defined")
    flush(stdout)

    # Add constraints to the model
    println("---------------")
    flush(stdout)


    # Each individual can only be assigned to one household
    @constraint(model, allocation * ones(n_hh,1) .<= 1)
    println("Constraint *Each individual can only be assigned to one household* is defined")
    flush(stdout)

    # Any individual added to a household makes it inhabited
    @constraint(model, [hh_id = 1:n_hh], household_inhabited[hh_id] .>= allocation[:, hh_id])
    println("Constraint *Any individual added to a household makes it inhabited* is defined")
    flush(stdout)

    # Any children added to the household makes it children inhabited
    @constraint(model, [hh_id = 1:n_hh], household_children_inhabited[hh_id] .>= allocation[:, hh_id] .* is_potential_child)
    println("Constraint *Any children added to the household makes it children inhabited* is defined")
    flush(stdout)

    # Any married female added to the household makes it inhabited
    @constraint(model, [hh_id = 1:n_hh], household_married_female_inhabited[hh_id] .>= allocation[:, hh_id] .* is_married_female)
    @constraint(model, household_married_female_inhabited .<= allocation' * is_married_female)
    println("Constraint *Any children added to the household makes it children inhabited* is defined")
    flush(stdout)

    # Any married male added to the household makes it inhabited
    @constraint(model, [hh_id = 1:n_hh], household_married_male_inhabited[hh_id] .>= allocation[:, hh_id] .* is_married_male)
    @constraint(model, household_married_male_inhabited .<= allocation' * is_married_male)
    println("Constraint *Any children added to the household makes it children inhabited* is defined")
    flush(stdout)

    # Any active household must have at least 1 adult
    
    println("Constraint *Any active household must have at least 1 adult* is defined")
    flush(stdout)

    # # Any individual added to a household makes it inhabited
    # @constraint(model, household_inhabited .>= allocation' * ones(n_indiv,1))
    # println("Constraint *Any individual added to a household makes it inhabited* is defined")
    # flush(stdout)

    # Any household must meet its capacity
    @constraint(model, allocation' * ones(n_indiv,1) .== hh_capacity .* household_inhabited)
    println("Constraint *Any household must meet its capacity* is defined")
    flush(stdout)

    # Any active household must have at least 1 adult
    @constraint(model, allocation' * is_adult .>= household_inhabited)
    println("Constraint *Any active household must have at least 1 adult* is defined")
    flush(stdout)

    # Any household cannot have more than 1 married male
    @constraint(model, allocation' * is_married_male .<= household_married_male_inhabited)
    println("Constraint *Any household cannot have more than 1 married male* is defined")
    flush(stdout)

    # Any household cannot have more than 1 married female
    @constraint(model, allocation' * is_married_female .<= household_married_female_inhabited)
    println("Constraint *Any household cannot have more than 1 married female* is defined")
    flush(stdout)

    # Children could be only in a household that has at least one parent
    @constraint(model, household_children_inhabited .<= household_married_female_inhabited + household_married_male_inhabited)
    println("Constraint *Children could be only in a household that has at least one parent* is defined")
    flush(stdout)

    # Number of children cannot exceed hh_capacity - parents
    @constraint(model, allocation' * is_potential_child .<= hh_capacity .- household_married_female_inhabited .- household_married_male_inhabited)
    println("Constraint *Number of children cannot exceed hh_capacity - parents* is defined")
    flush(stdout)

    # Optimize the model to find the best allocation of individuals to households
    optimize!(model)
    println("Objective value: ", objective_value(model))
    flush(stdout)

    
    # # Age difference between parents not bigger than 5 years
    # @constraint(model, [hh_id = 1:n_hh], sum(allocation[married_female_indices, hh_id] .* age_vector[married_female_indices]) - sum(allocation[married_male_indices, hh_id] .* age_vector[married_male_indices]) <= 5 + 100*(2 - household_married_male[hh_id] - household_married_female[hh_id] + penalty[hh_id]))
    # @constraint(model, [hh_id = 1:n_hh], sum(allocation[married_male_indices, hh_id] .* age_vector[married_male_indices]) - sum(allocation[married_female_indices, hh_id] .* age_vector[married_female_indices]) <= 5 + 100*(2 - household_married_male[hh_id] - household_married_female[hh_id] + penalty[hh_id]))
    # @constraint(model, [hh_id = 1:n_hh], sum(allocation[child_indices, hh_id]) .<= hh_capacity[hh_id] - household_married_female[hh_id] - household_married_male[hh_id])
    # println("Constraint *Age difference between parents not bigger than 5 years* is defined")
    # flush(stdout)

    # # Age difference between each child and parent must be more than 15 years and with at least one parent must be less than 40 years   
    # @constraint(model, [child_id in child_indices, hh_id in 1:n_hh, male_parent_id in married_male_indices],
    #     allocation[child_id, hh_id] * (age_vector[male_parent_id] - age_vector[child_id]) <= 40 + 100*(1-allocation[male_parent_id, hh_id] + male_parent_relaxation[child_id, hh_id]))

    # @constraint(model, [child_id in child_indices, hh_id in 1:n_hh, female_parent_id in married_female_indices],
    #     allocation[child_id, hh_id] * (age_vector[female_parent_id] - age_vector[child_id]) <= 40 + 100*(1-allocation[female_parent_id, hh_id] + female_parent_relaxation[child_id, hh_id]))

    # @constraint(model, [child_id in child_indices, hh_id in 1:n_hh, male_parent_id in married_male_indices],
    #     allocation[child_id, hh_id] * (age_vector[male_parent_id] - age_vector[child_id]) >= 15*allocation[child_id, hh_id] )

    # @constraint(model, [child_id in child_indices, hh_id in 1:n_hh, female_parent_id in married_female_indices],
    #     allocation[child_id, hh_id] * (age_vector[female_parent_id] - age_vector[child_id]) >= 15*allocation[child_id, hh_id] )


    # # If there is only one parent then no relaxation could be applied
    # @constraint(model, [child_id in child_indices, hh_id in 1:n_hh],
    # male_parent_relaxation[child_id, hh_id] + female_parent_relaxation[child_id, hh_id] <= household_married_female[hh_id])

    # @constraint(model, [child_id in child_indices, hh_id in 1:n_hh],
    # male_parent_relaxation[child_id, hh_id] + female_parent_relaxation[child_id, hh_id] <= household_married_male[hh_id])
    
    # println("Constraint *Age difference between each child and parent must be more than 15 years and with at least one parent must be less than 40 years* is defined")
    # flush(stdout)

    # Optimize the model to find the best allocation of individuals to households
    optimize!(model)
    println("Objective value: ", objective_value(model))
    flush(stdout)

    # Retrieve the allocation results from the model
    allocation_values = value.(allocation)
    household_inhabited = value.(household_inhabited)
    household_married_male_inhabited = value.(household_married_male_inhabited)
    household_married_female_inhabited = value.(household_married_female_inhabited)
    # penalty = value.(penalty)
    # female_parent_relaxation = value.(female_parent_relaxation)
    # male_parent_relaxation = value.(male_parent_relaxation)

    return allocation_values, household_inhabited, household_married_male_inhabited, household_married_female_inhabited, household_children_inhabited
end


"""
    disaggr_optimized_indiv(allocation_values, aggregated_individuals::DataFrame)

Disaggregate individuals into households based on allocation results.

# Arguments
- `allocation_values::Matrix{Float64}`: A matrix where each row corresponds to an individual and each column corresponds to a household, indicating whether the individual is allocated to the household.
- `aggregated_individuals::DataFrame`: A DataFrame containing the aggregated data of individuals, including identifiers and demographic details. 

# Returns
- `DataFrame`: A DataFrame containing disaggregated individuals.
"""
function disaggr_optimized_indiv(allocation_values::Matrix{Float64}, aggregated_individuals::DataFrame)
    # Initialize cumulative populations for disaggregation    
    cumulative_population_ind = cumsum(aggregated_individuals[!, POPULATION_COLUMN])
    individuals_count = sum(aggregated_individuals[!,POPULATION_COLUMN])
    # Disaggregate individuals based on allocation results
    disaggregated_individuals = DataFrame(
        id = 1:individuals_count,
        agg_ind_id = Vector{Union{Int,Missing}}(missing, individuals_count),
        household_id = Vector{Union{Int,Missing}}(missing, individuals_count),
    )
    for individual_id = 1:individuals_count

        # Assign individual ID to disaggregated_individuals
        agg_ind_id = findrow(cumulative_population_ind, individual_id)
        disaggregated_individuals[individual_id, :agg_ind_id] =
            aggregated_individuals[agg_ind_id, :id]

        # Assign household ID to disaggregated individuals
        household_id = findfirst(x -> x > 0.95, allocation_values[individual_id, :])
        if household_id === nothing
            disaggregated_individuals[individual_id, :household_id] = missing
        else
            disaggregated_individuals[individual_id, :household_id] = household_id
        end
    end
    return disaggregated_individuals
end  


"""
    disaggr_optimized_hh(allocation_values, aggregated_households, aggregated_individuals, parent_indices)

Disaggregates household data based on optimized allocation results, creating a new DataFrame of disaggregated households.

# Arguments
- `allocation_values::Matrix{Float64}`: A matrix indicating the optimized allocation of individuals to households.
- `aggregated_households::DataFrame`: A DataFrame containing aggregated household data.
- `aggregated_individuals::DataFrame`: A DataFrame containing individual data.
- `parent_indices::Vector{Int}`: A Vector of indices of individuals classified as parents.

# Returns
- `DataFrame`: A DataFrame representing disaggregated households.
"""
function disaggr_optimized_hh(allocation_values::Matrix{Float64}, aggregated_households::DataFrame, aggregated_individuals::DataFrame, parent_indices::Vector{Int})
    # Initialize cumulative populations for disaggregation 
    cumulative_population_hh = cumsum(aggregated_households[!, POPULATION_COLUMN])
    cumulative_population_ind = cumsum(aggregated_individuals[!, POPULATION_COLUMN])
    households_count = sum(aggregated_households[!,POPULATION_COLUMN])

    # Disaggregate households based on allocation results
    max_household_size = maximum(aggregated_households[:, HOUSEHOLD_SIZE_COLUMN])
    household_columns = [:agg_hh_id, :head_id, :partner_id]  # Initialize with parent columns
    for i = 1:(max_household_size-1)
        push!(household_columns, Symbol("child$(i)_id"))
    end
    disaggregated_households = DataFrame(id = 1:households_count)
    for column in household_columns
        disaggregated_households[!, column] =
            Vector{Union{Int,Missing}}(missing, households_count)
    end

    for household_id = 1:households_count

        # Add household ID from aggregated_households
        agg_hh_id = findfirst(x -> x >= household_id, cumulative_population_hh)
        disaggregated_households[household_id, :agg_hh_id] =
            aggregated_households[agg_hh_id, ID_COLUMN]

        # Assign parents and children
        assigned_individuals = findall(x -> x > 0.95, allocation_values[:, household_id])
        if length(assigned_individuals) == 1
            individual_id = findrow(cumulative_population_ind, assigned_individuals[1])
            disaggregated_households[household_id, :head_id] =
                aggregated_individuals[individual_id, :id]
        elseif length(assigned_individuals) >= 2
            println(household_id)
            println("----")
            parents = intersect(assigned_individuals, parent_indices)
            individual_id = findrow(cumulative_population_ind, parents[1])
            disaggregated_households[household_id, :head_id] =
                aggregated_individuals[individual_id, :id]
            if length(parents) == 2
                individual_id = findrow(cumulative_population_ind, parents[2])
                disaggregated_households[household_id, :partner_id] =
                    aggregated_individuals[individual_id, :id]
            end
            children = setdiff(assigned_individuals, parents)
            child_count = 0
            for child_id in children
                child_count += 1
                individual_id = findrow(cumulative_population_ind, child_id)
                disaggregated_households[household_id, Symbol("child$(child_count)_id")] =
                    aggregated_individuals[individual_id, :id]
            end
        end
    end
    return disaggregated_households
end