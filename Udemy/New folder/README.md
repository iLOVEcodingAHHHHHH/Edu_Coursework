# Installation instructions
Welcome to the Polars course!

- The current recommended python versions for the course are 3.11 or 3.12
- The bash and powershell scripts below assume `python` gives your current python 3 version. If this is not the case, try replacing `python` with `python3` in the scripts
- If you have any issues with the installation, let me know at liam@rhosignal.com

## Mac OS & Linux
- Unzip the polars-materials compressed file to a location on your system
- Open a terminal in this location
- Run the `pip-deploy.sh` script to download up-to-date versions of the required packages
- Activate the virtual environment with `source course_env/bin/activate`
- Confirm the virtual environment is activated with `which python`. The output should end with: `course_env/bin/python`
- Start the Jupyter notebooks with `jupyter lab`. This will either open the notebook in your browser or print a URL that you can use to open the notebook in your browser
- Check your installation by opening notebooks/TestInstallation.ipynb


## Windows
- Unzip the polars-materials compressed file to a directory on your system
- Open powershell in this directory
- Run the powershell script `create-course-environment.ps1`
- Run the powershell script `activate-course-environment.ps1`
- Confirm that the virtual environment is working (it should say course_env at the start of your powershell prompt)
- Run: `jupyter lab` 
- Check your installation by opening notebooks/TestInstallation.ipynb


