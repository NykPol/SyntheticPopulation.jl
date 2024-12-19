using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Revise
using DataFrames
using SyntheticPopulation




#each individual and each household represent 100.000 individuals or households
SCALE = 0.01 

#all values are based on China census data
individual_popoulation_size = 21890000


#individuals
marginal_ind_age_sex = DataFrame(
    sex = repeat(['M', 'F'], 18),
    age = repeat(2:5:87, inner = 2), 
    population = SCALE .* 10000 .* [52.6, 49.0, 48.5, 44.8, 33.6, 30.6, 34.6, 28.8, 71.6, 63.4, 99.6, 90.9, 130.9, 119.4, 110.8, 103.5, 83.8, 76.4, 84.2, 77.7, 84.2, 77.8, 82.8, 79.9, 67.7, 71.0, 56.9, 62.6, 31.5, 35.3, 18.5, 23.0, 15.2, 19.7, 12.5, 16.0]
    )

marginal_ind_sex_maritalstatus = DataFrame(
    sex = repeat(['M', 'F'], 4), 
    maritalstatus = repeat(["Never_married", "Married", "Divorced", "Widowed"], inner = 2), 
    population = SCALE .* [1679, 1611, 5859, 5774, 140, 206, 128, 426] ./ 0.00082
    )


marginal_ind_income = DataFrame(
    income = [25394, 44855, 63969, 88026, 145915], 
    population = repeat([individual_popoulation_size * SCALE / 5], 5)
    )

#households
household_total_population = 8230000
marginal_hh_size = DataFrame(
    hh_size = [1,2,3,4,5],
    population = Int.(round.(SCALE * household_total_population .* [0.299, 0.331, 0.217, 0.09, 0.063]))
    )

#generation of dataframe of individuals
aggregated_individuals = generate_joint_distribution(marginal_ind_age_sex, marginal_ind_sex_maritalstatus, marginal_ind_income, config_file = "tutorial_notebooks/config_file.json")
filter!(row -> row[SyntheticPopulation.POPULATION_COLUMN] >= 1, aggregated_individuals)
aggregated_individuals.id = 1:nrow(aggregated_individuals)
aggregated_individuals = add_indices_range_to_indiv(aggregated_individuals)
aggregated_individuals = add_individual_flags(aggregated_individuals)
adult_indices, married_male_indices, married_female_indices, parent_indices, child_indices, age_vector = prep_group_indices_for_indv_constraints(aggregated_individuals);



#generation of dataframe of households
aggregated_households = generate_joint_distribution(marginal_hh_size)
aggregated_households = add_indices_range_to_hh(aggregated_households)
hh_size1_indices, hh_size2_indices, hh_size3plus_indices, hh_capacity = prep_group_indices_for_hh_constraints(aggregated_households);

# Optimization
allocation_values, household_inhabited, household_married_male, household_married_female, penalty, female_parent_relaxation, male_parent_relaxation = define_and_run_optimization(aggregated_individuals
    , aggregated_households
    
    , hh_size1_indices
    , hh_size2_indices
    , hh_size3plus_indices
    , hh_capacity

    , adult_indices
    , married_male_indices
    , married_female_indices
    , parent_indices
    , child_indices
    , age_vector);  

disaggregated_individuals = disaggr_optimized_indiv(allocation_values, aggregated_individuals)
disaggregated_households = disaggr_optimized_hh(allocation_values, aggregated_households, aggregated_individuals, parent_indices)


# Define a function to perform the join for each role (head, partner, child1, etc.)
function join_individual_data(households_df, individuals_df, role_id::Symbol, suffix::String)

    # Perform the left join
    join_df = leftjoin(
        households_df,
        individuals_df,
        on=role_id => :id,
        makeunique=true,
        matchmissing = :equal
    )
    
    # Rename the joined columns with suffixes for clarity
    rename!(join_df, Dict(Symbol("maritalstatus") => Symbol("maritalstatus_$suffix"), 
                     Symbol("income") => Symbol("income_$suffix"),
                     Symbol("sex") => Symbol("sex_$suffix"),
                     Symbol("age") => Symbol("age_$suffix"),
                     Symbol("is_potential_parent") => Symbol("is_potential_parent_$suffix"),
                     Symbol("is_potential_child") => Symbol("is_potential_child_$suffix")))
    foreach(col -> join_df[!, col] = coalesce.(join_df[!, col], ""), names(join_df))
    join_df[!,Symbol("attributes_$suffix")] = string.(join_df[!,Symbol("sex_$suffix")], "|", join_df[!,Symbol("age_$suffix")], "|", join_df[!,Symbol("maritalstatus_$suffix")])
    return join_df
end

# Sequentially join for each role and add relevant data
disaggregated_households = join_individual_data(disaggregated_households, aggregated_individuals, :head_id, "head")
disaggregated_households = join_individual_data(disaggregated_households, aggregated_individuals, :partner_id, "partner")
disaggregated_households = join_individual_data(disaggregated_households, aggregated_individuals, :child1_id, "child1")
disaggregated_households = join_individual_data(disaggregated_households, aggregated_individuals, :child2_id, "child2")
disaggregated_households = join_individual_data(disaggregated_households, aggregated_individuals, :child3_id, "child3")
disaggregated_households = join_individual_data(disaggregated_households, aggregated_individuals, :child4_id, "child4")

disaggregated_households = leftjoin(
        disaggregated_households,
        aggregated_households,
        on=:agg_hh_id => :id,
        makeunique=true,
        matchmissing = :equal
    )

report_disaggregated_households = disaggregated_households[!, ["id", "hh_size", "attributes_head", "attributes_partner", "attributes_child1", "attributes_child2", "attributes_child3", "attributes_child4"]]
report_disaggregated_households[rand(1:nrow(report_disaggregated_households), 10),:]


using PyCall
using Colors
using Conda 
#Conda.add("folium")
folium = pyimport("folium")


#areas
URL = "https://osm-boundaries.com/Download/Submit?apiKey=2986553a70d2b1dd49788a148fcf2d22&db=osm20230102&osmIds=-2988894,-2988933,-2988895,-288600,-2988896,-2988946,-5505984,-2988897,-2988898,-2988899,-2988900,-5505985,-2988901,-2988902,-568660,-2988903&format=GeoJSON&srid=4326"
areas = SyntheticPopulation.generate_areas_dataframe_from_url(URL)

#aggregated_areas - population referenced from https://nj.tjj.beijing.gov.cn/nj/main/2021-tjnj/zk/indexeh.htm
aggregated_areas = copy(areas)
aggregated_areas.:population = SCALE .* 10000 .* [56.8, 313.2, 201.9, 345.1, 34.6, 184.0, 132.4, 45.7, 52.8, 39.3, 44.1, 131.3, 199.4, 226.9, 110.6, 70.9]
aggregated_areas.:population = round.(Int, aggregated_areas.population)
aggregated_areas


rename!(disaggregated_households, Symbol("agg_hh_id") => Symbol("hh_attr_id"))
# Create :individuals_allocated column
disaggregated_households[:, :individuals_allocated] = map(row -> any(x != "" for x in row), 
                                    eachrow(disaggregated_households[:, [:head_id, :partner_id, :child1_id, :child2_id, :child3_id, :child4_id]]))

disaggregated_households[!, [:head_id, :partner_id, :child1_id, :child2_id, :child3_id, :child4_id, :individuals_allocated]]
disaggregated_households = SyntheticPopulation.assign_areas_to_households!(disaggregated_households, aggregated_households, aggregated_areas)

m = folium.Map(location = [disaggregated_households.lat[1], disaggregated_households.lon[1]], zoom_start=11)
i = 1
for area in unique(disaggregated_households.area_id)
    colrs = distinguishable_colors(length(unique(disaggregated_households.area_id)), [RGB(1,0.6,0.5)])
    hh_color = "#$(hex(colrs[i]))"
    i += 1
    area  = filter(row -> row.area_id == area, disaggregated_households)
    for i in 1:nrow(area)
        folium.Circle(
            location = (area.lat[i], area.lon[i]),
            radius = 100,
            color = hh_color,
            fill = false,
            fill_color = hh_color
        ).add_to(m)
    end
end
m
m.save("map.html")


# Internal validation

# Prepare df after optimization for comparison
disaggregated_individuals_only_assigned = filter(row -> !ismissing(row.household_id), disaggregated_individuals)
disaggregated_individuals_only_assigned = leftjoin(disaggregated_individuals_only_assigned, aggregated_individuals[:, [:id, :maritalstatus, :age, :sex, :income]], on = :agg_ind_id => :id)
synthetic_aggregated_individuals = combine(groupby(disaggregated_individuals_only_assigned, [:maritalstatus, :age, :sex, :income]), nrow => :population)
rename!(synthetic_aggregated_individuals, Dict(:maritalstatus => :MARITAL_STATUS, :age => :AGE, :sex => :SEX, :income => :INCOME))

# Prepare IPF df to comparison
ipf_aggregated_individuals = copy(aggregated_individuals)
rename!(ipf_aggregated_individuals, Dict(:maritalstatus => :MARITAL_STATUS, :age => :AGE, :sex => :SEX, :income => :INCOME))
sum(ipf_aggregated_individuals[!,"population"])

# Compute contingency tables for IPF and POST_OPTIMIZATION populations
include("validation_notebooks/utils.jl")
synthetic_age_sex, synthetic_sex_marital, synthetic_income = compute_marginals(synthetic_aggregated_individuals)
ipf_age_sex, ipf_sex_marital, ipf_income = compute_marginals(ipf_aggregated_individuals)

# Validate AGE_SEX vs Synthetic
rename!(marginal_ind_age_sex, Dict(:age => :AGE, :sex => :SEX))
validate_table(ipf_age_sex, marginal_ind_age_sex)
validate_table(synthetic_age_sex, marginal_ind_age_sex)
sum(marginal_ind_age_sex[!,"population"])
sum(ipf_age_sex[!,"population"])

# Validate MARITAL_SEX vs Synthetic
rename!(marginal_ind_sex_maritalstatus, Dict(:maritalstatus => :MARITAL_STATUS, :sex => :SEX))
validate_table(ipf_sex_marital, marginal_ind_sex_maritalstatus)
validate_table(synthetic_sex_marital, marginal_ind_sex_maritalstatus)
sum(marginal_ind_sex_maritalstatus[!,"population"])
sum(ipf_sex_marital[!,"population"])

# Validate INCOME vs Synthetic
rename!(marginal_ind_income, Dict(:income => :INCOME))
validate_table(ipf_income, marginal_ind_income)
validate_table(synthetic_income, marginal_ind_income)
sum(marginal_ind_income[!,"population"])
sum(ipf_income[!,"population"])
