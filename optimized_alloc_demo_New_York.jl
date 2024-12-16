using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Revise
using DataFrames
using SyntheticPopulation

#each individual and each household represent 100.000 individuals or households
SCALE = 0.00003

#all values are based on China census data
individual_popoulation_size = 8258035

println("POPULATION: ", individual_popoulation_size * SCALE)

#individuals
population = [
    # male
    227349
    ,223900
    ,244087
    ,228152
    ,237353
    ,315374
    ,347733
    ,295564
    ,270094
    ,237024
    ,244597
    ,236294
    ,246669
    ,201415
    ,162392
    ,112733
    ,68463
    ,61740
    # female
    ,218215
    ,213700
    ,233291
    ,222591
    ,253345
    ,347152
    ,353146
    ,304273
    ,275160
    ,256192
    ,264607
    ,268286
    ,264165
    ,237136
    ,205842
    ,160794
    ,111421
    ,107786
]


marginal_ind_age_sex = DataFrame(
    sex = repeat(['M', 'F'], inner=18),
    age = repeat(2:5:87, outer=2), 
    population = SCALE * population
    )
    
marginal_ind_sex_maritalstatus = DataFrame(
    sex = repeat(['M', 'F'], 4), 
    maritalstatus = repeat(["Never_married", "Married", "Divorced", "Widowed"], inner = 2), 
    population = SCALE .* [1469519, 1536292, 1515237, 1470917, 205732, 341398, 75108, 286919]
    )

# 0.835 is a share of pop ages > [2, 7, 12]
marginal_ind_income = DataFrame(
    income = [9999, 14999, 24999, 34999, 49999, 64999, 74999, 99999, 100000], 
    population = Int.(round.(SCALE * individual_popoulation_size * 0.835 .* [0.014, 0.014, 0.05, 0.093, 0.16, 0.139, 0.074, 0.144, 0.311]))
    )

#households
household_total_population = 3394750
marginal_hh_size = DataFrame(
    hh_size = [1,2,3,4],
    population = Int.(round.(SCALE * household_total_population .* [0.345, 0.297, 0.15, 0.209]))
    )

#generation of dataframe of individuals
aggregated_individuals = generate_joint_distribution(marginal_ind_sex_maritalstatus, marginal_ind_income, marginal_ind_age_sex, config_file = "tutorial_notebooks/config_file_NY.json")
filter!(row -> row[SyntheticPopulation.POPULATION_COLUMN] >= 1, aggregated_individuals)
aggregated_individuals.id = 1:nrow(aggregated_individuals)
aggregated_individuals = add_indices_range_to_indiv(aggregated_individuals)
aggregated_individuals = add_individual_flags(aggregated_individuals)
disaggregated_ipf_individuals = disaggr_ipf_individuals(aggregated_individuals)              


#generation of dataframe of households
aggregated_households = generate_joint_distribution(marginal_hh_size)
aggregated_households = add_indices_range_to_hh(aggregated_households)
hh_capacity = reduce(vcat, [fill(size, length(r)) for (size, r) in zip(aggregated_households[!, SyntheticPopulation.HOUSEHOLD_SIZE_COLUMN], aggregated_households[!, "hh_indices"])])

allocation_values, household_inhabited, household_married_male_inhabited, household_married_female_inhabited, household_children_inhabited = define_and_run_optimization(disaggregated_ipf_individuals, hh_capacity);  


# -----HELPER
# Example data
# indiv_household = [1 0; 1 0; 1 0; 0 1; 0 1; 0 1; 1 0; 1 0; 0 1; 0 1]  # 10x2 matrix (individual to household)
# adult = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]  # 10x1 vector (0 = not adult, 1 = adult)

# indiv_household' * ones(10,1)
# # Matrix multiplication: Multiply the indiv_household matrix (10x2) with the adult vector (10x1)
# adult_sum_by_household = indiv_household * ones(2,1)  # Result is a 2x1 vector
# indiv_household
# # Output the result
# println("Sum of adults in each household: ", adult_sum_by_household)
#---------

filter(row -> coalesce(row.household_id, 999) == 999, disaggregated_individuals)
disaggregated_individuals = disaggr_optimized_indiv(allocation_values, aggregated_individuals)
disaggregated_households = disaggr_optimized_hh(allocation_values, aggregated_households, aggregated_individuals, filter(row -> row.is_potential_parent == true, disaggregated_ipf_individuals)[!, "id"])
disaggregated_households
value.(household_married_female_inhabited)[54]
value.(household_married_male_inhabited)[54]

filter(row -> coalesce(row.household_id, 999) == 54, disaggregated_individuals)

filter(row -> row.id == 66, aggregated_individuals)
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
#disaggregated_households = join_individual_data(disaggregated_households, aggregated_individuals, :child4_id, "child4")

disaggregated_households = leftjoin(
        disaggregated_households,
        aggregated_households,
        on=:agg_hh_id => :id,
        makeunique=true,
        matchmissing = :equal
    )

report_disaggregated_households = disaggregated_households[!, ["id", "hh_size", "attributes_head", "attributes_partner", "attributes_child1", "attributes_child2", "attributes_child3"]]#, "attributes_child4"
report_disaggregated_households[rand(1:nrow(report_disaggregated_households), 10),:]



using PyCall
using Colors
using Conda 
#Conda.add("folium")
folium = pyimport("folium")


#areas
URL = "https://osm-boundaries.com/Download/Submit?apiKey=2986553a70d2b1dd49788a148fcf2d22&db=osm20230102&osmIds=-2552450,-369518,-2552485,-369519,-962876&format=GeoJSON&srid=4326"
areas = SyntheticPopulation.generate_areas_dataframe_from_url(URL)
aggregated_areas = copy(areas)
aggregated_areas.:population = SCALE .* [1356476, 2561225, 1597451, 2252196, 490687]
aggregated_areas.:population = round.(Int, aggregated_areas.population)

rename!(disaggregated_households, Symbol("agg_hh_id") => Symbol("hh_attr_id"))
# Create :individuals_allocated column
disaggregated_households[:, :individuals_allocated] = map(row -> any(x != "" for x in row), 
                                    eachrow(disaggregated_households[:, [:head_id, :partner_id, :child1_id, :child2_id, :child3_id]]))
                                    disaggregated_households
disaggregated_households[!, [:head_id, :partner_id, :child1_id, :child2_id, :child3_id, :individuals_allocated]]
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













