echo "In this script we create a virtual environment for the Polars course and install the required packages"
# Create a virtual environment
echo "Create virtual environment"
python -m venv course_env
# Activate the virtual environment
echo "Activate the virtual environment, upgrade pip and install the packages into the virtual environment"
source course_env/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
echo "Finished installing the packages, now activate the environment with: source course_env/bin/activate"
