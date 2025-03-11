import pandas as pd
import numpy as np
import os

def get_project_root():
    """Get the absolute path to the project root directory"""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
    project_root = os.path.dirname(current_dir)  # up one level to project root
    return project_root

class InvestmentAnalysis:
    def __init__(self):
        """Initialize with updated CAPEX values"""
        # Investment costs (CAPEX) per MW
        self.capex = {
            'wind': 1400000,     # CHF/MW
            'solar': 900000,     # CHF/MW
            'battery1': 250000,  # CHF/MW
            'battery2': 450000,  # CHF/MWy
        }
        
        # Financial parameters
        self.discount_rate = 0.08  # 8%
        self.time_horizons = [10, 20, 30]  # Years to calculate NPV for
        
        # Technical lifetime of assets
        self.lifetime = {
            'wind': 19,
            'solar': 25,
            'battery1': 6,
            'battery2': 8,
        }
        
        self.annual_opex_percent = {
            'wind': 0.04,
            'solar': 0.02,
            'battery1': 0.03,
            'battery2': 0.04,
        }

        # Add load factors for variants
        self.load_factors = {
            'low': 0.8,
            'nominal': 1.0,
            'high': 1.2
        }

    def calculate_initial_investment(self, installed_capacity):
        """Calculate initial investment based on installed capacities"""
        investment = 0
        for tech, capacity in installed_capacity.items():
            # Convert technology name to match capex keys if needed
            tech_key = tech.lower()  # Convert to lowercase to match capex keys
            if tech_key in self.capex:
                investment += capacity * self.capex[tech_key]
                print(f"Adding investment for {tech_key}: {capacity} MW * ${self.capex[tech_key]}/MW = ${capacity * self.capex[tech_key]}")
        
        print(f"Total initial investment: ${investment}")
        return investment

    def calculate_annual_costs(self, operational_costs, installed_capacity):
        """Calculate total annual costs including O&M"""
        annual_cost = operational_costs  # From scenario results
        
        # Add maintenance costs
        for tech, capacity in installed_capacity.items():
            if tech in self.annual_opex_percent:
                maintenance = capacity * self.capex[tech] * self.annual_opex_percent[tech]
                annual_cost += maintenance
        
        return annual_cost

    def calculate_npv(self, initial_investment, annual_costs, years, installed_capacity):
        """Calculate NPV for a specific time horizon"""
        npv = -initial_investment
        for year in range(years):
            # Add replacement costs if asset lifetime is exceeded
            replacement_cost = 0
            if year > 0:  # Check for replacements
                for tech, lifetime in self.lifetime.items():
                    if year % lifetime == 0:  # Time to replace
                        capacity = installed_capacity.get(tech, 0)
                        replacement_cost += capacity * self.capex[tech]
            
            yearly_cashflow = -annual_costs - replacement_cost
            npv += yearly_cashflow / (1 + self.discount_rate)**(year + 1)
        return npv

    def calculate_annuity(self, npv, years):
        """Calculate annuity payment for a specific time horizon"""
        if npv >= 0:  # For positive NPV, return 0 (no payments needed)
            return 0
        annuity_factor = (self.discount_rate * (1 + self.discount_rate)**years) / \
                        ((1 + self.discount_rate)**years - 1)
        return -npv * annuity_factor  # Negative NPV becomes positive annuity

    def analyze_scenario(self, scenario_results_path, master_gen_path, run_sensitivity=False):
        """Main analysis function"""
        try:
            # Get project root for path resolution
            project_root = get_project_root()
            
            # Resolve absolute paths
            scenario_results_path = os.path.join(project_root, 'data', 'results', 'scenario_results.csv')
            working_dir = os.path.join(project_root, 'data', 'working')
            scenarios_params_path = os.path.join(working_dir, 'scenarios_parameters.csv')
            
            # Load data
            scenario_results = pd.read_csv(scenario_results_path)
            scenarios_params = pd.read_csv(scenarios_params_path)
            
            results_list = []  # Change to list to store multiple variants
            for scenario in scenario_results['base_scenario'].unique():
                print(f"\n{'='*50}")
                print(f"Processing scenario: {scenario}")
                
                # Get scenario configuration
                scenario_config = scenarios_params[scenarios_params['scenario_name'] == scenario].iloc[0]
                gen_positions = eval(scenario_config['gen_positions'])
                storage_positions = eval(scenario_config['storage_units'])
                
                # Count installed capacity
                installed_capacity = {}
                for _, gen_type in gen_positions.items():
                    tech_key = gen_type.lower()
                    installed_capacity[tech_key] = installed_capacity.get(tech_key, 0) + 1
                
                for _, storage_type in storage_positions.items():
                    installed_capacity[storage_type] = installed_capacity.get(storage_type, 0) + 1
                
                # Calculate base values
                initial_inv = self.calculate_initial_investment(installed_capacity)
                
                # Determine which variants to process based on sensitivity flag
                variants_to_process = {'nominal': 1.0}
                if run_sensitivity:
                    variants_to_process.update({'low': 0.8, 'high': 1.2})
                
                # Process each variant
                for variant, load_factor in variants_to_process.items():
                    scenario_id = f"{scenario}_{variant}"
                    print(f"\nProcessing variant: {variant} (load factor: {load_factor})")
                    
                    # Get scenario data for the variant
                    scenario_data = scenario_results[
                        (scenario_results['base_scenario'] == scenario) & 
                        (scenario_results['variant'] == 'nominal')
                    ].iloc[0]
                    
                    # Scale costs based on load factor
                    base_annual_cost = float(scenario_data['annual_cost'])
                    scaled_annual_cost = base_annual_cost * load_factor
                    annual_costs = self.calculate_annual_costs(scaled_annual_cost, installed_capacity)
                    
                    # Calculate NPV and annuity for different time horizons
                    npv_results = {}
                    annuity_results = {}
                    for years in self.time_horizons:
                        npv = self.calculate_npv(initial_inv, annual_costs, years, installed_capacity)
                        annuity = self.calculate_annuity(npv, years)
                        npv_results[f'npv_{years}y'] = npv
                        annuity_results[f'annuity_{years}y'] = annuity
                    
                    # Scale generation values
                    gen_results = {}
                    for col in scenario_data.index:
                        if col.startswith('gen_'):
                            base_value = float(scenario_data[col])
                            gen_results[col] = base_value * load_factor
                    
                    # Compile results for this variant
                    variant_results = {
                        'scenario_id': scenario_id,
                        'base_scenario': scenario,
                        'variant': variant,
                        'load_factor': load_factor,
                        'installed_capacity': str(installed_capacity),
                        'initial_investment': initial_inv,
                        'annual_cost': scaled_annual_cost,
                        'annual_costs': annual_costs,
                        **npv_results,
                        **annuity_results,
                        **gen_results,
                        'scenario_name': scenario
                    }
                    
                    results_list.append(variant_results)
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results_list)
            
            # Save results
            output_path = os.path.join(project_root, 'data', 'results', 'scenario_results_with_investment.csv')
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")
            
            return results_df
        
        except Exception as e:
            print(f"Error in analyze_scenario: {str(e)}")
            raise

# If running this file directly
if __name__ == '__main__':
    project_root = get_project_root()
    analysis = InvestmentAnalysis()
    
    # Use proper paths relative to project root
    results = analysis.analyze_scenario(
        os.path.join(project_root, 'data', 'results', 'scenario_results.csv'),
        os.path.join(project_root, 'data', 'working', 'master_gen.csv'),
        run_sensitivity=True  # Default to True when running directly
    )
    
    print(results)
    
    # Format results for display
    display_df = results.copy()
    
    # Format monetary values
    monetary_columns = ['initial_investment', 'annual_costs'] + \
                      [f'npv_{y}y' for y in analysis.time_horizons] + \
                      [f'annuity_{y}y' for y in analysis.time_horizons]
    
    for col in monetary_columns:
        display_df[col] = display_df[col].map('${:,.2f}'.format)
    
    # Format installed capacity
    display_df['installed_capacity'] = display_df['installed_capacity'].apply(
        lambda x: '\n'.join([f"{k}: {v:.2f} MW" for k, v in x.items()])
    )
    
    # Sort by 30-year NPV and get top 10
    top_10 = display_df.sort_values('npv_30y', ascending=True).head(10)
    
    print("\n=== Top 10 Scenarios by 30-year NPV ===")
    print("\nDetailed Results:")
    for idx, row in top_10.iterrows():
        print(f"\nScenario: {idx}")
        print("Installed Capacity:")
        print(row['installed_capacity'])
        print(f"Initial Investment: {row['initial_investment']}")
        print(f"Annual Costs: {row['annual_costs']}")
        print("\nNPV Analysis:")
        for years in analysis.time_horizons:
            print(f"{years}-year NPV: {row[f'npv_{years}y']}")
            print(f"{years}-year Annuity: {row[f'annuity_{years}y']}")
        print("-" * 50)
    
    # Alternative: Create an Excel file with formatted results
    writer = pd.ExcelWriter('data/results/investment_analysis.xlsx', engine='xlsxwriter')
    
    # Write to Excel with formatting
    display_df.to_excel(writer, sheet_name='All Results')
    top_10.to_excel(writer, sheet_name='Top 10 Scenarios')
    
    # Get workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['All Results']
    
    # Add formatting
    money_format = workbook.add_format({'num_format': '$#,##0.00'})
    wrap_format = workbook.add_format({'text_wrap': True})
    
    # Set column widths
    worksheet.set_column('B:B', 40, wrap_format)  # Installed capacity
    worksheet.set_column('C:F', 15, money_format)  # Money columns
    
    writer.close()
    
    print("\nResults have been saved to 'data/results/investment_analysis.xlsx'")
    
    # Get raw results sorted by NPV
    sorted_results = results.sort_values('npv', ascending=True)
    
    # Get specific metrics
    npv_series = sorted_results['npv']
    annuities = sorted_results['annuity'] 