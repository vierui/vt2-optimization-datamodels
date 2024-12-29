import os

def update_readme_with_scenarios():
    # Read the template
    with open('README.md', 'r') as f:
        template = f.read()
    
    # Get list of scenario folders
    results_dir = 'data/results'
    scenario_folders = [f for f in os.listdir(results_dir) 
                       if os.path.isdir(os.path.join(results_dir, f)) 
                       and not f.startswith('.')]
    
    # Generate markdown links for each scenario
    scenario_links = []
    for folder in sorted(scenario_folders):
        analysis_file = f"{folder}_analysis.md"
        if os.path.exists(os.path.join(results_dir, folder, analysis_file)):
            scenario_links.append(f"- [{folder}](data/results/{folder}/{analysis_file})")
    
    # Replace placeholder with generated links
    readme_content = template.replace("${scenario_links}", "\n".join(scenario_links))
    
    # Write updated README
    with open('README.md', 'w') as f:
        f.write(readme_content)

if __name__ == "__main__":
    update_readme_with_scenarios() 