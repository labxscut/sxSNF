

#!/bin/bash

# Clean up auxiliary files in the current directory
echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

# Clean up auxiliary files in the parent directory
cd ..
rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

echo "Cleanup complete!"
