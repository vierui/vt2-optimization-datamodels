import pandas as pd
import numpy as np

class InvestmentAnalysis:
    def __init__(self):
        # Investment costs (CAPEX) per MW
        self.capex = {
            'wind': 1200000,    # $/MW
            'solar': 800000,    # $/MW
            'battery1': 250000, # $/MW
            'battery2': 200000, # $/MW
        }
        
        # Financial parameters
        self.discount_rate = 0.08  # 8%
        self.time_horizons = [10, 20, 30]  # Years to calculate NPV for
        
        # Technical lifetime of assets
        self.lifetime = {
            'wind': 25,
            'solar': 25,
            'battery1': 8,
            'battery2': 8,
        }
        
        self.annual_opex_percent = {
            'wind': 0.02,
            'solar': 0.02,
            'battery1': 0.03,
            'battery2': 0.03,
        }

    def calculate_initial_investment(self, installed_capacity):
        """Calculate initial investment based on installed capacities"""
        investment = 0
        for tech, capacity in installed_capacity.items():
            if tech in self.capex:
                investment += capacity * self.capex[tech]
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

    def calculate_npv(self, initial_investment, annual_costs, years):
        """Calculate NPV for a specific time horizon"""
        npv = -initial_investment
        for year in range(years):
            # Add replacement costs if asset lifetime is exceeded
            replacement_cost = 0
            if year > 0:  # Check for replacements
                for tech, lifetime in self.lifetime.items():
                    if year % lifetime == 0:  # Time to replace
                        capacity = self.installed_capacity.get(tech, 0)
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

    def analyze_scenario(self, scenario_results_path, master_gen_path):
        """Main analysis function"""
        # Load scenario results and master_gen data
        scenario_results = pd.read_csv(scenario_results_path)
        master_gen = pd.read_csv(master_gen_path)
        
        # Get unique first timestamp to extract installed capacities
        first_timestamp = master_gen['time'].iloc[0]
        
        # Extract installed capacities
        installed_capacity = {
            'wind': float(master_gen[(master_gen['time'] == first_timestamp) & 
                                   (master_gen['type'] == 'wind')]['pmax'].iloc[0]),
            'solar': float(master_gen[(master_gen['time'] == first_timestamp) & 
                                    (master_gen['type'] == 'solar')]['pmax'].iloc[0]),
            'battery1': float(master_gen[(master_gen['time'] == first_timestamp) & 
                                       (master_gen['id'] == 101)]['pmax'].iloc[0]),
            'battery2': float(master_gen[(master_gen['time'] == first_timestamp) & 
                                       (master_gen['id'] == 102)]['pmax'].iloc[0])
        }
        
        results = {}
        for scenario in scenario_results['scenario_name'].unique():
            scenario_data = scenario_results[scenario_results['scenario_name'] == scenario]
            
            # Store installed capacity for replacement calculations
            self.installed_capacity = installed_capacity
            
            # Calculate initial investment
            initial_inv = self.calculate_initial_investment(installed_capacity)
            
            # Calculate annual costs
            annual_costs = self.calculate_annual_costs(
                float(scenario_data['annual_cost'].iloc[0]), 
                installed_capacity
            )
            
            # Calculate NPV and annuity for different time horizons
            npv_results = {}
            annuity_results = {}
            
            for years in self.time_horizons:
                npv = self.calculate_npv(initial_inv, annual_costs, years)
                annuity = self.calculate_annuity(npv, years)
                
                npv_results[f'npv_{years}y'] = npv
                annuity_results[f'annuity_{years}y'] = annuity
            
            results[scenario] = {
                'installed_capacity': installed_capacity,
                'initial_investment': initial_inv,
                'annual_costs': annual_costs,
                **npv_results,
                **annuity_results
            }
            
        return pd.DataFrame(results).T

# Usage example
if __name__ == "__main__":
    analysis = InvestmentAnalysis()
    results = analysis.analyze_scenario(
        'data/results/scenario_results.csv',
        'data/working/master_gen.csv'
    )
    
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