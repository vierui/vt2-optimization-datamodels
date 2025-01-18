# Create auxiliary directory
$aux_dir = "out";
# Also output PDF to the main directory
$out_dir = "out";
ensure_path($aux_dir);

# Copy PDF back to main directory
$success_cmd = 'cp out/main.pdf .';

# Clean up auxiliary files but keep PDF
$cleanup_mode = 1; 